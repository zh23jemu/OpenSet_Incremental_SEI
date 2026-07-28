"""生成 ADS-B 目标簇数补齐分裂的配对三种子验证计划。

计划采用严格配对设计：每个 seed 都包含 baseline_locked 与 target_split
两组运行。seed31 复用已完成的 Job 44670427；seed7/13 在当前 Slurm job
中复用既有长序列 closed-set checkpoint，只改变目标分裂开关。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHECKPOINTS = {
    7: "results/stage4/adsb_main_long_radcil_seed7_44470736/closedset_adsb_90known_strict.pth",
    13: "results/stage4/adsb_main_long_radcil_seed13_44470736/closedset_adsb_90known_strict.pth",
    31: "results/stage4/adsb_main_long_radcil_seed31_44467424/closedset_adsb_90known_strict.pth",
}

REUSED_SEED31_DIRS = {
    "baseline_locked": "results/stage4/adsb_target_split_seed31_baseline_locked_44670427",
    "target_split": "results/stage4/adsb_target_split_seed31_target_split_44670427",
}


def experiment_args(
    data_root: str,
    checkpoint: str,
    save_dir: str,
    seed: int,
    enable_target_split: bool,
) -> list[str]:
    """构造单次 ADS-B Long-RADCIL 验证参数。

    baseline 与 target_split 共用同一数据协议、checkpoint、训练后端和
    MV-ACC 锁定参数；target_split 只额外打开协议目标簇数补齐分裂。
    """

    argv = [
        "--data_root", data_root,
        "--save_dir", save_dir,
        "--checkpoint", checkpoint,
        "--initial_known_classes", "90",
        "--round_size", "10",
        "--num_rounds", "3",
        "--seed", str(seed),
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
    if enable_target_split:
        argv.extend([
            "--enable_mvacc_target_split",
            "--mvacc_target_split_clusters", "10",
            "--mvacc_target_split_max_added", "4",
            "--mvacc_target_split_silhouette", "0.26",
        ])
    return argv


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """生成包含 3 seeds x 2 variants 的审计计划。"""

    runs = []
    for seed in (7, 13, 31):
        for variant, enable_target_split in (
            ("baseline_locked", False),
            ("target_split", True),
        ):
            if seed == 31:
                save_dir = REUSED_SEED31_DIRS[variant]
                source = "reused_job_44670427"
            else:
                save_dir = f"{args.output_prefix}_seed{seed}_{variant}_{args.job_id}"
                source = "current_job"
            runs.append({
                "seed": seed,
                "variant": variant,
                "save_dir": save_dir,
                "source": source,
                "checkpoint": CHECKPOINTS[seed],
                "overrides": {
                    "enable_mvacc_target_split": enable_target_split,
                    "mvacc_target_split_clusters": 10 if enable_target_split else 0,
                    "mvacc_target_split_max_added": 4 if enable_target_split else 0,
                    "mvacc_target_split_silhouette": 0.26 if enable_target_split else None,
                },
                "argv": experiment_args(
                    args.data_root,
                    CHECKPOINTS[seed],
                    save_dir,
                    seed,
                    enable_target_split,
                ),
            })

    return {
        "schema_version": "stage4_adsb_target_split_multiseed_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": args.job_id,
        "seeds": [7, 13, 31],
        "paired_control": "baseline_locked",
        "candidate": "target_split",
        "target_split": {
            "target_clusters": 10,
            "max_added": 4,
            "silhouette": 0.26,
        },
        "locked_backend": {
            "backbone": "adsb_long",
            "old_new_batch_ratio": 2.0,
            "replay_weight": 3.0,
        },
        "selection_boundary": (
            "跨种子决策优先比较 discovery 侧最终簇数是否接近 round_size=10、"
            "Label-free Silhouette、Raw HDBSCAN Confidence Mean、Cluster Size CV "
            "和 Small Cluster Fraction；真实标签聚类指标与 held-out 增量准确率"
            "只作为事后审计。"
        ),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 ADS-B target split 三种子配对计划。")
    parser.add_argument("--data-root", default="数据集/ADS-B/Dataset")
    parser.add_argument("--output-prefix", default="results/stage4/adsb_target_split_multiseed")
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", default="results/stage4/stage4_adsb_target_split_multiseed_plan.json")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_plan(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ADS-B target split 三种子计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
