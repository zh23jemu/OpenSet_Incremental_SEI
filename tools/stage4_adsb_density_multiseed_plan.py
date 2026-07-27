"""生成 ADS-B 密度比例候选的配对三种子确认计划。"""

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


def experiment_args(data_root: str, checkpoint: str, save_dir: str, seed: int, ratio: float) -> list[str]:
    """构造固定 Long-RADCIL 后端的单次运行参数。"""
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
        "--mvacc_min_cluster_ratio", str(ratio),
        "--mvacc_merge_threshold", "0.74",
        "--mvacc_split_size_factor", "1.75",
        "--mvacc_split_silhouette", "0.30",
        "--disable_visualization",
    ]


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """seed7/13 新运行配对变体，seed31 复用已完成的配对结果。"""
    runs = []
    for seed in (7, 13, 31):
        for variant, ratio in (("baseline_locked", 0.03), ("density_ratio_0p02", 0.02)):
            if seed == 31:
                save_dir = f"results/stage4/adsb_discovery_ablation_seed31_{variant}_44474297"
                source = "reused_job_44474297"
            else:
                save_dir = f"{args.output_prefix}_seed{seed}_{variant}_{args.job_id}"
                source = "current_job"
            runs.append({
                "seed": seed,
                "variant": variant,
                "ratio": ratio,
                "save_dir": save_dir,
                "source": source,
                "argv": experiment_args(args.data_root, CHECKPOINTS[seed], save_dir, seed, ratio),
            })
    return {
        "schema_version": "stage4_adsb_density_multiseed_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": args.job_id,
        "seeds": [7, 13, 31],
        "candidate": "mvacc_min_cluster_ratio=0.02",
        "paired_control": "mvacc_min_cluster_ratio=0.03",
        "locked_backend": {"backbone": "adsb_long", "old_new_batch_ratio": 2.0, "replay_weight": 3.0},
        "selection_boundary": (
            "跨种子决策优先比较 discovery 侧 silhouette、HDBSCAN 置信度、簇大小 CV 和净簇数变化；"
            "真实标签聚类指标与增量准确率只作为事后效果审计。"
        ),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 ADS-B 密度比例配对三种子计划。")
    parser.add_argument("--data-root", default="数据集/ADS-B/Dataset")
    parser.add_argument("--output-prefix", default="results/stage4/adsb_density_multiseed")
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", default="results/stage4/stage4_adsb_density_multiseed_plan.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_plan(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ADS-B 密度比例三种子计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
