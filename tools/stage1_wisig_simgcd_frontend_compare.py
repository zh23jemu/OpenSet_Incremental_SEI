"""阶段 1 WiSig 真实特征上的发现前端对比。

该脚本把 `utils.simgcd_discovery_adapter.SimGCDStyleDiscoveryAdapter`
接入 WiSig strict 10+10x3 协议的 frozen embeddings，并与同一轮次、
同一 frozen backbone 下的 Deep-HDBSCAN 和 MV-ACC 发现结果比较。

无泄漏边界：
- `DiscoveryAdapterInput` 只接收 Day1 已知类训练/验证特征与当前轮次
  discovery 未标注特征；
- 当前轮次真实标签只在 `predict` 完成后用于事后 NMI/ARI/purity/
  Hungarian Acc 报告；
- held-out evaluation split 不进入该脚本的发现前端选择或训练。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 允许 Slurm 直接从项目根目录或 tools 目录启动脚本，不要求手动设置 PYTHONPATH。
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp_wisig_mvacc_cil_strict import (  # noqa: E402
    TRAINING_RECIPE_VERSION,
    calibrate_mvacc_on_known_day1,
    clustering_metrics,
    extract_deep_features,
    load_closedset_model,
    load_wisig_crossday_10known_3round,
    run_discovery,
    save_csv,
    select_classic_feature_view,
    save_selection,
    set_seed,
    stratified_train_validation_split,
    to_dataset,
    train_closedset_model,
)
from utils.discovery_adapter_contract import DiscoveryAdapterInput, validate_discovery_output  # noqa: E402
from utils.simgcd_discovery_adapter import SimGCDStyleDiscoveryAdapter  # noqa: E402


def make_mvacc_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """构造可复用 `run_discovery` 的最小参数集合。

    `run_discovery` 来自现有 WiSig strict 实验脚本，内部会访问较多
    聚类、图融合、可视化和可靠性参数。这里集中补齐默认值，避免在
    新脚本中复制 MV-ACC 的实现细节。
    """
    return argparse.Namespace(
        save_dir=str(args.save_dir),
        seed=int(args.seed),
        alpha=float(args.alpha),
        top_k=int(args.top_k),
        graph_dim=int(args.graph_dim),
        adaptive_fusion=bool(args.adaptive_fusion),
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        cflcg_local_k=int(args.cflcg_local_k),
        cflcg_extractor=None,
        cflcg_gate_open=True,
        min_cluster_size=int(args.min_cluster_size),
        min_samples=int(args.min_samples),
        reliability_min_cluster_size=10,
        reliability_min_prob=0.30,
        reliability_threshold=0.50,
        merge_threshold=0.70,
        no_mutual_nearest=False,
        iter_merge_threshold=0.86,
        iter_merge_lambda_deep=0.4,
        iter_merge_lambda_rf=0.2,
        iter_merge_lambda_graph=0.4,
        nodrop_merge_rounds=5,
        nodrop_merge_threshold=0.78,
        nodrop_assign_lambda_deep=0.4,
        nodrop_assign_lambda_rf=0.2,
        nodrop_assign_lambda_graph=0.4,
        mvacc_min_cluster_ratio=float(args.mvacc_min_cluster_ratio),
        mvacc_min_samples=int(args.mvacc_min_samples),
        mvacc_merge_rounds=int(args.mvacc_merge_rounds),
        mvacc_merge_threshold=float(args.mvacc_merge_threshold),
        mvacc_calibration_ratios=str(args.mvacc_calibration_ratios),
        mvacc_calibration_thresholds=str(args.mvacc_calibration_thresholds),
        mvacc_lambda_deep=float(args.mvacc_lambda_deep),
        mvacc_lambda_rf=float(args.mvacc_lambda_rf),
        mvacc_lambda_graph=float(args.mvacc_lambda_graph),
        mvacc_prototypes_per_class=5,
        mvacc_split_rounds=int(args.mvacc_split_rounds),
        mvacc_split_size_factor=float(args.mvacc_split_size_factor),
        mvacc_split_min_part_ratio=float(args.mvacc_split_min_part_ratio),
        mvacc_split_silhouette=float(args.mvacc_split_silhouette),
        enable_visualization=False,
        save_detailed_visualizations=False,
        visualization_method="pca",
    )


def checkpoint_metadata(args: argparse.Namespace, dataset_path: Path, selected_rx_list: list[int]) -> dict[str, Any]:
    """生成与主 WiSig strict 脚本一致的 checkpoint 元数据。"""
    return {
        "dataset_path": os.path.abspath(dataset_path),
        "selected_rx_list": list(selected_rx_list),
        "known_tx": list(range(args.initial_known_classes)),
        "training_day_index": 0,
        "development_ratio": float(args.development_ratio),
        "sample_split_protocol": "stratified_random_60_10_30_v1",
        "seed": int(args.seed),
        "use_supcon": bool(args.use_supcon),
        "supcon_weight": float(args.supcon_weight),
        "supcon_temperature": float(args.supcon_temperature),
        "training_recipe_version": TRAINING_RECIPE_VERSION,
        "closedset_validation_fraction": float(args.closedset_val_ratio),
        "rf_augmentation": bool(not args.disable_rf_augmentation),
        "supcon_projection_dim": int(args.projection_dim) if args.use_supcon else 0,
        "supcon_projection_hidden_dim": int(args.projection_hidden_dim) if args.use_supcon else 0,
    }


def simgcd_row(
    args: argparse.Namespace,
    round_name: str,
    day_name: str,
    known_train_z: np.ndarray,
    known_train_y: np.ndarray,
    val_z: np.ndarray,
    val_y: np.ndarray,
    round_z: np.ndarray,
    round_y: np.ndarray,
) -> dict[str, Any]:
    """运行 SimGCD 式适配器并生成与聚类表兼容的结果行。"""
    adapter = SimGCDStyleDiscoveryAdapter(
        temperature=float(args.simgcd_temperature),
        random_state=int(args.seed),
        n_init=int(args.simgcd_n_init),
    )
    data = DiscoveryAdapterInput(
        known_features=known_train_z,
        known_labels=known_train_y,
        validation_features=val_z,
        validation_labels=val_y,
        unlabeled_features=round_z,
        round_name=round_name,
        expected_new_classes=int(args.round_size) if args.use_protocol_expected_k else None,
    )
    adapter.fit(data)
    output = adapter.predict(data)
    validate_discovery_output(output, expected_samples=round_z.shape[0])
    metrics = clustering_metrics(round_y, output.cluster_labels)
    return {
        "Method": "SimGCD-style prototype head",
        "Round": round_name,
        "Discovery Day": day_name,
        "True New Classes": int(args.round_size),
        "Samples": int(round_y.shape[0]),
        "Initial Cluster Count": int(output.estimated_new_classes),
        "Final Cluster Count": int(metrics["Clusters"]),
        "Cluster Count Error": int(abs(int(metrics["Clusters"]) - int(args.round_size))),
        "Noise": float(metrics["Noise"]),
        "Assignment Coverage": 1.0,
        "Purity": float(metrics["Purity"]),
        "NMI": float(metrics["NMI"]),
        "ARI": float(metrics["ARI"]),
        "Hungarian Acc": float(metrics["Hungarian Acc"]),
        "Confidence Mean": float(np.mean(output.confidence)),
        "Confidence Min": float(np.min(output.confidence)),
        "Confidence Max": float(np.max(output.confidence)),
        "Adapter Estimated New Classes": int(output.estimated_new_classes),
        "Uses Unknown True Labels": False,
        "Uses Eval Set": False,
        "Diagnostics": json.dumps(output.diagnostics, ensure_ascii=False, sort_keys=True),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 1 WiSig SimGCD 式发现前端真实特征对比。")
    parser.add_argument("--dataset_path", required=True, help="WiSig CrossDay PKL 路径。")
    parser.add_argument("--save_dir", default="results/stage1/wisig_simgcd_frontend_compare", help="输出目录。")
    parser.add_argument("--checkpoint", default=None, help="闭集 checkpoint；缺省为 save_dir/closedset_rx2_day1_10known.pth。")
    parser.add_argument("--train_closedset", action="store_true", help="强制重新训练初始闭集模型。")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--use_supcon", action="store_true")
    parser.add_argument("--supcon_weight", type=float, default=0.1)
    parser.add_argument("--supcon_temperature", type=float, default=0.2)
    parser.add_argument("--closedset_val_ratio", type=float, default=1.0 / 7.0)
    parser.add_argument("--projection_hidden_dim", type=int, default=128)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--disable_rf_augmentation", action="store_true")
    parser.add_argument("--disable_cflcg", action="store_true")
    parser.add_argument("--cflcg_gate_threshold", type=float, default=0.80)
    parser.add_argument("--cflcg_local_k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial_known_classes", type=int, default=10)
    parser.add_argument("--round_size", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--development_ratio", type=float, default=0.70)
    parser.add_argument("--selected_rx_list", type=str, default="2")
    parser.add_argument("--min_cluster_size", type=int, default=10)
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--graph_dim", type=int, default=16)
    parser.add_argument("--adaptive_fusion", action="store_true")
    parser.add_argument("--alpha_min", type=float, default=0.65)
    parser.add_argument("--alpha_max", type=float, default=0.95)
    parser.add_argument("--mvacc_min_cluster_ratio", type=float, default=0.045)
    parser.add_argument("--mvacc_min_samples", type=int, default=5)
    parser.add_argument("--mvacc_merge_rounds", type=int, default=12)
    parser.add_argument("--mvacc_merge_threshold", type=float, default=0.74)
    parser.add_argument("--disable_mvacc_calibration", action="store_true")
    parser.add_argument("--mvacc_calibration_ratios", type=str, default="0.03,0.045,0.06,0.08")
    parser.add_argument("--mvacc_calibration_thresholds", type=str, default="0.74,0.78,0.82,0.86")
    parser.add_argument("--mvacc_lambda_deep", type=float, default=0.5)
    parser.add_argument("--mvacc_lambda_rf", type=float, default=0.1)
    parser.add_argument("--mvacc_lambda_graph", type=float, default=0.4)
    parser.add_argument("--mvacc_split_rounds", type=int, default=6)
    parser.add_argument("--mvacc_split_size_factor", type=float, default=1.75)
    parser.add_argument("--mvacc_split_min_part_ratio", type=float, default=0.30)
    parser.add_argument("--mvacc_split_silhouette", type=float, default=0.30)
    parser.add_argument("--simgcd_temperature", type=float, default=0.1)
    parser.add_argument("--simgcd_n_init", type=int, default=10)
    parser.add_argument("--use_protocol_expected_k", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.initial_known_classes != 10 or args.round_size != 10 or args.num_rounds != 3:
        raise ValueError("当前阶段 1 对比脚本固定用于 WiSig 10+10x3 strict 协议。")
    selected_rx_list = [int(x.strip()) for x in args.selected_rx_list.split(",") if x.strip()]
    if selected_rx_list != [2]:
        raise ValueError("阶段 1 WiSig 对比保持与短实验一致：--selected_rx_list 2。")

    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path).resolve()
    if args.checkpoint is None:
        args.checkpoint = str(save_dir / "closedset_rx2_day1_10known.pth")
    mvacc_args = make_mvacc_namespace(args)

    splits = load_wisig_crossday_10known_3round(
        dataset_path=str(dataset_path),
        transpose_to_model=True,
        train_ratio=float(args.development_ratio),
        selected_rx_list=selected_rx_list,
        seed=int(args.seed),
    )
    train_x = splits["day1_known_train"]["X"]
    train_y = splits["day1_known_train"]["y"]
    train_set = to_dataset(train_x, train_y)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metadata = checkpoint_metadata(args, dataset_path, selected_rx_list)
    if args.train_closedset or not os.path.exists(args.checkpoint):
        model = train_closedset_model(
            train_set,
            args.initial_known_classes,
            args.feat_dim,
            args.epochs,
            args.batch_size,
            args.lr,
            device,
            args.checkpoint,
            use_supcon=args.use_supcon,
            supcon_weight=args.supcon_weight,
            supcon_temperature=args.supcon_temperature,
            checkpoint_metadata=metadata,
            seed=args.seed,
            validation_fraction=args.closedset_val_ratio,
            use_rf_augmentation=not args.disable_rf_augmentation,
            projection_hidden_dim=args.projection_hidden_dim,
            projection_dim=args.projection_dim,
        )
    else:
        model = load_closedset_model(
            args.checkpoint,
            args.initial_known_classes,
            args.feat_dim,
            device,
            expected_metadata=metadata,
        )

    train_z, train_y = extract_deep_features(model, train_x, train_y, args.test_batch_size, device)
    day1_training_subset, day1_validation_subset = stratified_train_validation_split(
        train_set, validation_fraction=args.closedset_val_ratio, seed=args.seed
    )
    train_idx = np.asarray(day1_training_subset.indices, dtype=np.int64)
    val_idx = np.asarray(day1_validation_subset.indices, dtype=np.int64)

    mvacc_args.cflcg_extractor = None
    mvacc_args.cflcg_gate_open = True
    if not args.disable_cflcg:
        cflcg_selection, cflcg_extractor = select_classic_feature_view(
            train_x[val_idx], train_y[val_idx], args.cflcg_gate_threshold
        )
        mvacc_args.cflcg_extractor = cflcg_extractor
        mvacc_args.cflcg_gate_open = bool(cflcg_selection["rf_gate_open"])
        save_selection(cflcg_selection, str(save_dir))

    if not args.disable_mvacc_calibration:
        calibrate_mvacc_on_known_day1(train_x[val_idx], train_y[val_idx], train_z[val_idx], mvacc_args)

    rows: list[dict[str, Any]] = []
    round_keys = ["day2_unknown_round1", "day3_unknown_round2", "day4_unknown_round3"]
    for round_index, split_key in enumerate(round_keys, start=1):
        round_name = f"R{round_index}"
        round_x = splits[split_key]["X"]
        round_y = splits[split_key]["y"]
        day_name = splits[split_key].get("day", f"Day {round_index + 1}")
        round_z, round_y = extract_deep_features(model, round_x, round_y, args.test_batch_size, device)

        deep_row, _ = run_discovery(
            "Deep only", round_name, day_name, args.round_size, round_x, round_y, round_z, mvacc_args
        )
        deep_row["Method"] = "Deep-HDBSCAN"
        rows.append(deep_row)

        mvacc_row, _ = run_discovery(
            "MV-ACC", round_name, day_name, args.round_size, round_x, round_y, round_z, mvacc_args
        )
        rows.append(mvacc_row)

        rows.append(
            simgcd_row(
                args=args,
                round_name=round_name,
                day_name=day_name,
                known_train_z=train_z[train_idx],
                known_train_y=train_y[train_idx],
                val_z=train_z[val_idx],
                val_y=train_y[val_idx],
                round_z=round_z,
                round_y=round_y,
            )
        )

    csv_path = save_dir / "frontend_comparison.csv"
    save_csv(rows, str(csv_path))
    summary = {
        "schema_version": "stage1_wisig_simgcd_frontend_compare_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "checkpoint": str(args.checkpoint),
        "seed": int(args.seed),
        "selected_rx_list": selected_rx_list,
        "protocol": "WiSig strict 10+10x3, Day1 60/10/30, current-round discovery only",
        "leakage_boundary": {
            "adapter_inputs": "known train/validation embeddings + current round unlabeled embeddings",
            "hidden_labels": "posthoc metrics only",
            "heldout_eval": "not used",
        },
        "csv": str(csv_path),
        "rows": rows,
    }
    json_path = save_dir / "frontend_comparison_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"阶段 1 WiSig SimGCD 前端对比 CSV：{csv_path}")
    print(f"阶段 1 WiSig SimGCD 前端对比 JSON：{json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
