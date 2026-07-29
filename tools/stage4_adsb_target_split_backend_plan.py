"""生成 ADS-B target split 后端遗忘对照计划。

本计划用于收敛阶段 4 的剩余风险：target split 已能缓解 ADS-B R3
欠聚类，但 MV-ACC-CIL 的 Forgetting 仍高于 DOI-style。这里固定
target split 发现前端，只重新打开同协议 CIL baselines，用来判断低遗忘
优势是否来自 DOI-style 后端本身，而不是旧的欠聚类发现条件。
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


def experiment_args(data_root: str, checkpoint: str, save_dir: str, seed: int) -> list[str]:
    """构造单个 seed 的严格实验参数。

    注意这里不传 ``--disable_cil_baselines``，从而让共享 MV-ACC 发现后端
    和 Deep-HDBSCAN 端到端后端表一起产出。target split 的阈值仍采用
    已完成三种子消融中表现最稳定的默认候选，不在本计划中继续搜索。
    """

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
        "--enable_mvacc_target_split",
        "--mvacc_target_split_clusters", "10",
        "--mvacc_target_split_max_added", "4",
        "--mvacc_target_split_silhouette", "0.26",
        "--disable_visualization",
    ]


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """生成三种子 target split 后端 baseline 对照计划。"""

    runs = []
    for seed in (7, 13, 31):
        save_dir = f"{args.output_prefix}_seed{seed}_{args.job_id}"
        checkpoint = CHECKPOINTS[seed]
        runs.append({
            "seed": seed,
            "save_dir": save_dir,
            "checkpoint": checkpoint,
            "argv": experiment_args(args.data_root, checkpoint, save_dir, seed),
        })

    return {
        "schema_version": "stage4_adsb_target_split_backend_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": args.job_id,
        "seeds": [7, 13, 31],
        "output_prefix": args.output_prefix,
        "risk_question": (
            "在 ADS-B target split 缓解欠聚类后，DOI-style 的低遗忘优势是否仍然存在，"
            "以及 MV-ACC-CIL 的 Forgetting 风险是否能被发现前端补丁一起收敛。"
        ),
        "locked_discovery": {
            "target_split": True,
            "target_clusters": 10,
            "max_added": 4,
            "silhouette": 0.26,
            "min_cluster_ratio": 0.03,
            "merge_threshold": 0.74,
        },
        "locked_backend": {
            "backbone": "adsb_long",
            "old_new_batch_ratio": 2.0,
            "replay_weight": 3.0,
            "cil_baselines_enabled": True,
        },
        "decision_rule": (
            "若 target split 下 MV-ACC-CIL 同时提升 Overall 且 Forgetting 接近 DOI-style，"
            "则 ADS-B 遗忘风险可进一步收敛；若 DOI-style 仍显著低遗忘但 Overall/New 偏低，"
            "则风险归因到 RADCIL 与稳定原型后端的旧/新权衡，不再继续小阈值搜索。"
        ),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 ADS-B target split 后端遗忘对照计划。")
    parser.add_argument("--data-root", default="数据集/ADS-B/Dataset")
    parser.add_argument("--output-prefix", default="results/stage4/adsb_target_split_backend_baselines")
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", default="results/stage4/stage4_adsb_target_split_backend_plan.json")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_plan(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ADS-B target split 后端对照计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
