"""生成 ADS-B seed31 协议目标簇数补齐分裂验证计划。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def common_args(data_root: str, checkpoint: str, save_dir: str) -> list[str]:
    """固定 ADS-B 表征和 RADCIL 后端，只让目标分裂开关成为唯一变量。"""
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


def target_split_args(argv: list[str], max_added: int, silhouette: float) -> list[str]:
    """在基线参数后追加目标簇数补齐分裂参数，保持其它参数完全相同。"""
    return [
        *argv,
        "--enable_mvacc_target_split",
        "--mvacc_target_split_clusters", "10",
        "--mvacc_target_split_max_added", str(max_added),
        "--mvacc_target_split_silhouette", str(silhouette),
    ]


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """构造可审计计划；候选选择边界只允许使用 discovery 侧无标签结构指标。"""
    baseline_dir = f"{args.output_prefix}_baseline_locked_{args.job_id}"
    target_dir = f"{args.output_prefix}_target_split_{args.job_id}"
    baseline_argv = common_args(args.data_root, args.checkpoint, baseline_dir)
    target_argv = target_split_args(
        common_args(args.data_root, args.checkpoint, target_dir),
        max_added=args.max_added,
        silhouette=args.silhouette,
    )
    return {
        "schema_version": "stage4_adsb_target_split_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": args.job_id,
        "seed": 31,
        "checkpoint": args.checkpoint,
        "locked_backend": {"backbone": "adsb_long", "old_new_batch_ratio": 2.0, "replay_weight": 3.0},
        "target_split": {
            "target_clusters": 10,
            "max_added": int(args.max_added),
            "silhouette": float(args.silhouette),
        },
        "selection_boundary": (
            "本验证只比较 baseline_locked 与 target_split。候选是否值得扩 seed7/13，"
            "优先看 discovery 侧 Label-free Silhouette、Raw HDBSCAN Confidence Mean、"
            "Cluster Size CV、Small Cluster Fraction 和最终簇数是否接近协议目标；"
            "真实标签指标和增量准确率仅作事后审计。"
        ),
        "runs": [
            {"name": "baseline_locked", "save_dir": baseline_dir, "overrides": {}, "argv": baseline_argv},
            {
                "name": "target_split",
                "save_dir": target_dir,
                "overrides": {
                    "enable_mvacc_target_split": True,
                    "mvacc_target_split_clusters": 10,
                    "mvacc_target_split_max_added": int(args.max_added),
                    "mvacc_target_split_silhouette": float(args.silhouette),
                },
                "argv": target_argv,
            },
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 ADS-B seed31 目标簇数补齐分裂验证计划。")
    parser.add_argument("--data-root", default="数据集/ADS-B/Dataset")
    parser.add_argument(
        "--checkpoint",
        default="results/stage4/adsb_main_long_radcil_seed31_44467424/closedset_adsb_90known_strict.pth",
    )
    parser.add_argument("--output-prefix", default="results/stage4/adsb_target_split_seed31")
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--max-added", type=int, default=4)
    parser.add_argument("--silhouette", type=float, default=0.26)
    parser.add_argument("--output", default="results/stage4/stage4_adsb_target_split_plan.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_plan(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ADS-B 目标分裂验证计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
