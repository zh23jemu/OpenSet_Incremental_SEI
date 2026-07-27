"""生成 ADS-B 无标签轮次自适应密度三种子验证计划。"""

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

BASELINE_DIRS = {
    7: "results/stage4/adsb_density_multiseed_seed7_baseline_locked_44474381",
    13: "results/stage4/adsb_density_multiseed_seed13_baseline_locked_44474381",
    31: "results/stage4/adsb_discovery_ablation_seed31_baseline_locked_44474297",
}


def experiment_args(data_root: str, checkpoint: str, save_dir: str, seed: int) -> list[str]:
    """固定 Long-RADCIL 后端，只启用预注册的无标签密度门控。"""

    return [
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
        "--enable_mvacc_adaptive_density",
        "--mvacc_adaptive_candidate_ratio", "0.02",
        "--mvacc_adaptive_max_added_clusters", "2",
        "--mvacc_adaptive_min_silhouette_gain", "0.04",
        "--mvacc_adaptive_max_confidence_drop", "0.03",
        "--mvacc_adaptive_max_cv_increase", "0.10",
        "--mvacc_adaptive_max_small_fraction_increase", "0.10",
        "--mvacc_adaptive_min_partition_agreement", "0.50",
        "--disable_visualization",
    ]


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    runs = []
    for seed in (7, 13, 31):
        save_dir = f"{args.output_prefix}_seed{seed}_{args.job_id}"
        runs.append({
            "seed": seed,
            "save_dir": save_dir,
            "baseline_dir": BASELINE_DIRS[seed],
            "checkpoint": CHECKPOINTS[seed],
            "argv": experiment_args(args.data_root, CHECKPOINTS[seed], save_dir, seed),
        })
    return {
        "schema_version": "stage4_adsb_adaptive_density_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": args.job_id,
        "seeds": [7, 13, 31],
        "selection_boundary": (
            "每轮只使用 discovery 聚类的 silhouette、HDBSCAN 置信度、簇大小 CV、"
            "小簇比例、候选间 ARI 和相对新增簇数；真实标签与 held-out 指标仅事后审计。"
        ),
        "anchor_ratio": 0.03,
        "candidate_ratio": 0.02,
        "locked_backend": {
            "backbone": "adsb_long",
            "old_new_batch_ratio": 2.0,
            "replay_weight": 3.0,
        },
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 ADS-B 自适应密度三种子计划。")
    parser.add_argument("--data-root", default="数据集/ADS-B/Dataset")
    parser.add_argument("--output-prefix", default="results/stage4/adsb_adaptive_density")
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", default="results/stage4/stage4_adsb_adaptive_density_plan.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_plan(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ADS-B 自适应密度三种子计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
