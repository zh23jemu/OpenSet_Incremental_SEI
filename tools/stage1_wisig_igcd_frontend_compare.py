"""阶段 1 WiSig 真实特征上的 IGCD strict 最小前端对比。

该脚本复用 `stage1_wisig_simgcd_frontend_compare.py` 中已经通过 Slurm
验证的数据加载、闭集 checkpoint、Deep-HDBSCAN 与 MV-ACC 对比逻辑，
只额外接入 `IGCDMinimalDiscoveryAdapter`。这样可以在同一 WiSig
strict 10+10x3 discovery 轮次上检查 IGCD-style 最小入口是否具备
真实 RF frozen embeddings baseline 价值。

无泄漏边界：
- fit/predict 只接收 Day1 已知类训练/验证 embedding 与当前轮未标注
  discovery embedding；
- 当前轮真实标签只在 predict 后用于事后 NMI/ARI/purity/Hungarian Acc；
- held-out evaluation split 不参与发现前端拟合、类别数选择或置信度校准。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 允许 Slurm 从项目根目录直接运行脚本，同时不要求手动设置 PYTHONPATH。
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.stage1_wisig_simgcd_frontend_compare import (  # noqa: E402
    calibrate_mvacc_on_known_day1,
    checkpoint_metadata,
    clustering_metrics,
    extract_deep_features,
    load_closedset_model,
    load_wisig_crossday_10known_3round,
    make_mvacc_namespace,
    parse_args as parse_shared_args,
    run_discovery,
    save_csv,
    save_selection,
    select_classic_feature_view,
    set_seed,
    stratified_train_validation_split,
    to_dataset,
    train_closedset_model,
)
from utils.discovery_adapter_contract import DiscoveryAdapterInput, validate_discovery_output  # noqa: E402
from utils.igcd_minimal_adapter import IGCDMinimalDiscoveryAdapter  # noqa: E402


def igcd_row(
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
    """运行 IGCD strict 最小适配器并返回统一聚类指标行。"""
    adapter = IGCDMinimalDiscoveryAdapter(
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
        "Method": "IGCD-minimal prototype discovery",
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


def main() -> int:
    """运行真实 WiSig frozen embeddings 上的 IGCD 前端对比。"""
    args = parse_shared_args()
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
            igcd_row(
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
        "schema_version": "stage1_wisig_igcd_frontend_compare_v1",
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
        "implementation_level": "IGCD minimal strict entry, not full paper reproduction",
        "csv": str(csv_path),
        "rows": rows,
    }
    json_path = save_dir / "frontend_comparison_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"阶段 1 WiSig IGCD 前端对比 CSV：{csv_path}")
    print(f"阶段 1 WiSig IGCD 前端对比 JSON：{json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
