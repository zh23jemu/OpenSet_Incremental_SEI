"""生成 ADS-B seed31 发现前端单因素消融的审计计划。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VARIANTS = (
    ("baseline_locked", {}),
    ("density_ratio_0p02", {"mvacc_min_cluster_ratio": 0.02}),
    ("merge_threshold_0p86", {"mvacc_merge_threshold": 0.86}),
    ("split_relaxed", {"mvacc_split_size_factor": 1.50, "mvacc_split_silhouette": 0.20}),
)


def common_args(data_root: str, checkpoint: str, save_dir: str) -> list[str]:
    """固定表征、训练后端和数据协议，只允许变体覆盖一个发现因素。"""
    return [
        "--data_root", data_root,
        "--save_dir", save_dir,
        "--checkpoint", checkpoint,
        "--initial_known_classes", "90",
        "--round_size", "10",
        "--num_rounds", "3",
        "--seed", "31",
        "--backbone", "adsb_long",
        "--batch_size", "128",
        "--test_batch_size", "256",
        "--disable_cil_baselines",
        "--use_supcon",
        "--supcon_weight", "0.1",
        "--cil_head_warmup_epochs", "2",
        "--cil_joint_epochs", "8",
        "--cil_replay_weight", "3.0",
        "--radcil_old_new_batch_ratio", "2.0",
        "--disable_mvacc_calibration",
        "--mvacc_min_cluster_ratio", "0.03",
        "--mvacc_merge_threshold", "0.74",
        "--mvacc_split_size_factor", "1.75",
        "--mvacc_split_silhouette", "0.30",
        "--disable_visualization",
    ]


def apply_overrides(argv: list[str], overrides: dict[str, float]) -> list[str]:
    """在固定参数表中原位替换单因素值，保证计划 JSON 可直接审计。"""
    result = list(argv)
    for key, value in overrides.items():
        flag = f"--{key}"
        pos = result.index(flag)
        result[pos + 1] = str(value)
    return result


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    runs = []
    for name, overrides in VARIANTS:
        save_dir = f"{args.output_prefix}_{name}_{args.job_id}"
        argv = apply_overrides(common_args(args.data_root, args.checkpoint, save_dir), overrides)
        runs.append({"name": name, "save_dir": save_dir, "overrides": overrides, "argv": argv})
    return {
        "schema_version": "stage4_adsb_discovery_ablation_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": args.job_id,
        "seed": 31,
        "checkpoint": args.checkpoint,
        "locked_backend": {"backbone": "adsb_long", "old_new_batch_ratio": 2.0, "replay_weight": 3.0},
        "selection_boundary": (
            "候选比较只使用 discovery 侧 Label-free Silhouette、Raw HDBSCAN Confidence Mean、"
            "Cluster Size CV 和初始到最终簇损失；真实标签聚类指标与增量准确率仅作事后报告。"
        ),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 ADS-B seed31 发现单因素消融计划。")
    parser.add_argument("--data-root", default="数据集/ADS-B/Dataset")
    parser.add_argument(
        "--checkpoint",
        default="results/stage4/adsb_main_long_radcil_seed31_44467424/closedset_adsb_90known_strict.pth",
    )
    parser.add_argument("--output-prefix", default="results/stage4/adsb_discovery_ablation_seed31")
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", default="results/stage4/stage4_adsb_discovery_ablation_plan.json")
    args = parser.parse_args()
    plan = build_plan(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ADS-B 发现消融计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
