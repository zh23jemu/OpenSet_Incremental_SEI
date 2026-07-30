"""Stage 11 训练期跨天表征适配 discovery-only 入口。"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cross_day_representation_adaptation import adapt_model_cross_day


def _load_profile(args):
    if args.dataset_profile == "lora25":
        from experiments import exp_wisig_mvacc_cil_strict as exp

        splits = exp.load_lora25_diffdays_3round(args.dataset_path)
        model = exp.load_closedset_model(
            args.checkpoint,
            10,
            args.feat_dim,
            args.device,
            expected_metadata=None,
        )
        train_key = "day1_backbone_train"
        round_keys = [
            "day2_unknown_round1",
            "day3_unknown_round2",
            "day4_unknown_round3",
        ]
        round_size = 5
    else:
        from experiments import exp_adsb_mvacc_cil_strict as exp

        splits = exp.load_adsb_90known_3round(
            args.data_root,
            initial_known_classes=90,
            round_size=10,
            development_ratio=0.70,
            known_selection_seed=args.seed,
        )
        model = exp.load_closedset_model(
            args.checkpoint,
            90,
            args.feat_dim,
            args.device,
            backbone="adsb_long",
            expected_metadata=None,
        )
        train_key = "day1_known_train"
        round_keys = [
            "day2_unknown_round1",
            "day3_unknown_round2",
            "day4_unknown_round3",
        ]
        round_size = 10
    return exp, splits, model, train_key, round_keys, round_size


def _build_runtime_args(seed: int) -> SimpleNamespace:
    """构造 build_round_features 所需的固定、无真值参数。"""

    return SimpleNamespace(
        alpha=0.50,
        top_k=15,
        graph_dim=16,
        seed=int(seed),
        adaptive_fusion=False,
        alpha_min=0.25,
        alpha_max=0.75,
        cflcg_local_k=12,
        cflcg_extractor=None,
        cflcg_gate_open=True,
        discovery_feature_adapter="none",
    )


def _run_one(args, method: str, exp, splits, model, train_key, round_keys, round_size):
    known_x = splits[train_key]["X"]
    known_y = splits[train_key]["y"]
    rows = []
    model_seed = copy.deepcopy(model)
    runtime_args = _build_runtime_args(args.seed)
    for index, round_key in enumerate(round_keys, start=1):
        round_x = splits[round_key]["X"]
        round_y = splits[round_key]["y"]
        if method == "none":
            adapted_model = model_seed
            diagnostics = {
                "Cross-Day Adapter": "none",
                "Cross-Day Epochs": 0,
            }
        else:
            result = adapt_model_cross_day(
                model_seed,
                known_x,
                known_y,
                round_x,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                consistency_weight=args.consistency_weight,
                coral_weight=args.coral_weight,
                seed=args.seed + index,
            )
            adapted_model = result.model
            diagnostics = result.diagnostics

        # 特征提取函数需要一个占位 y 参数来保持现有接口，但 discovery-only
        # 阶段不能把当前轮真实标签用于任何训练或聚类决策，因此这里明确传入
        # 全零占位数组，真实标签只在最后的离线审计指标中读取。
        discovery_placeholder_y = np.zeros(len(round_x), dtype=np.int64)
        z_round, _ = exp.extract_deep_features(
            adapted_model,
            round_x,
            discovery_placeholder_y,
            args.test_batch_size,
            args.device,
        )
        feats = exp.build_round_features(
            round_x,
            z_round,
            runtime_args,
            cflcg_mode="local",
        )
        gpcc = exp.run_gpcc(feats, target_clusters=round_size, seed=args.seed)
        metrics = exp.clustering_metrics(round_y, gpcc.labels)
        rows.append({
            "Method": f"Stage11-{method}",
            "Round": f"R{index}",
            "True New Classes": int(round_size),
            "Final Cluster Count": int(metrics["Clusters"]),
            "NMI": float(metrics["NMI"]),
            "ARI": float(metrics["ARI"]),
            "Purity": float(metrics["Purity"]),
            "Hungarian Acc": float(metrics["Hungarian Acc"]),
            **diagnostics,
            **gpcc.diagnostics,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 11 训练期跨天表征适配 discovery-only。")
    parser.add_argument("--dataset_profile", choices=["lora25", "adsb"], required=True)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--method", choices=["none", "cross_day"], default="cross_day")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--consistency_weight", type=float, default=0.5)
    parser.add_argument("--coral_weight", type=float, default=0.05)
    args = parser.parse_args()
    if args.dataset_profile == "lora25" and not args.dataset_path:
        raise ValueError("--dataset_path is required for lora25.")
    if args.dataset_profile == "adsb" and not args.data_root:
        raise ValueError("--data_root is required for adsb.")
    args.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    exp, splits, model, train_key, round_keys, round_size = _load_profile(args)
    rows = _run_one(args, args.method, exp, splits, model, train_key, round_keys, round_size)
    output = Path(args.save_dir) / "clustering_results.csv"
    exp.save_csv(rows, str(output))
    (Path(args.save_dir) / "stage11_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
