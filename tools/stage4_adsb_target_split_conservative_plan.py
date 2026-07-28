"""生成 ADS-B 保守 target split 参数消融计划。

目标不是重新证明 target split 可行，而是解决三种子结果暴露出的剩余风险：
默认 ``max_added=4, silhouette=0.26`` 能稳定补齐 R3 欠聚类，但 seed7/13
新类收益不稳定且簇大小 CV 上升。因此本计划只比较更保守的补齐策略。
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

REUSED_DIRS = {
    (7, "baseline_locked"): "results/stage4/adsb_target_split_multiseed_seed7_baseline_locked_44766923",
    (7, "target_split_default"): "results/stage4/adsb_target_split_multiseed_seed7_target_split_44766923",
    (13, "baseline_locked"): "results/stage4/adsb_target_split_multiseed_seed13_baseline_locked_44766923",
    (13, "target_split_default"): "results/stage4/adsb_target_split_multiseed_seed13_target_split_44766923",
    (31, "baseline_locked"): "results/stage4/adsb_target_split_seed31_baseline_locked_44670427",
    (31, "target_split_default"): "results/stage4/adsb_target_split_seed31_target_split_44670427",
}

VARIANTS = (
    {
        "name": "baseline_locked",
        "source": "reused",
        "enable_target_split": False,
        "max_added": 0,
        "silhouette": None,
    },
    {
        "name": "target_split_default",
        "source": "reused",
        "enable_target_split": True,
        "max_added": 4,
        "silhouette": 0.26,
    },
    {
        "name": "target_split_max2_s026",
        "source": "current_job",
        "enable_target_split": True,
        "max_added": 2,
        "silhouette": 0.26,
    },
    {
        "name": "target_split_m4_s034",
        "source": "current_job",
        "enable_target_split": True,
        "max_added": 4,
        "silhouette": 0.34,
    },
    {
        "name": "target_split_m4_s038",
        "source": "current_job",
        "enable_target_split": True,
        "max_added": 4,
        "silhouette": 0.38,
    },
)


def experiment_args(
    data_root: str,
    checkpoint: str,
    save_dir: str,
    seed: int,
    max_added: int,
    silhouette: float,
) -> list[str]:
    """构造保守 target split 单次运行参数。

    所有候选固定 ADS-B strict 协议、长序列 checkpoint、Long-RADCIL 后端、
    MV-ACC ratio=0.03 和基础 split 参数，只改变目标补齐的强度。
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
        "--enable_mvacc_target_split",
        "--mvacc_target_split_clusters", "10",
        "--mvacc_target_split_max_added", str(max_added),
        "--mvacc_target_split_silhouette", str(silhouette),
        "--disable_visualization",
    ]


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """生成 3 seeds x 5 variants 的完整审计计划。"""

    runs = []
    for seed in (7, 13, 31):
        for variant in VARIANTS:
            name = str(variant["name"])
            if variant["source"] == "reused":
                save_dir = REUSED_DIRS[(seed, name)]
                source = "reused_job_44766923_or_44670427"
                argv = []
            else:
                save_dir = f"{args.output_prefix}_seed{seed}_{name}_{args.job_id}"
                source = "current_job"
                argv = experiment_args(
                    args.data_root,
                    CHECKPOINTS[seed],
                    save_dir,
                    seed,
                    int(variant["max_added"]),
                    float(variant["silhouette"]),
                )
            runs.append({
                "seed": seed,
                "variant": name,
                "save_dir": save_dir,
                "source": source,
                "checkpoint": CHECKPOINTS[seed],
                "overrides": {
                    "enable_mvacc_target_split": bool(variant["enable_target_split"]),
                    "mvacc_target_split_max_added": variant["max_added"],
                    "mvacc_target_split_silhouette": variant["silhouette"],
                },
                "argv": argv,
            })

    return {
        "schema_version": "stage4_adsb_target_split_conservative_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": args.job_id,
        "seeds": [7, 13, 31],
        "baseline": "baseline_locked",
        "default_candidate": "target_split_default",
        "conservative_candidates": [
            "target_split_max2_s026",
            "target_split_m4_s034",
            "target_split_m4_s038",
        ],
        "selection_boundary": (
            "优先用无标签结构指标选择候选：九轮绝对簇误差应低于 baseline，"
            "簇大小 CV 和小簇比例不应比默认 target split 系统性恶化；"
            "held-out 标签指标只做事后审计。"
        ),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 ADS-B 保守 target split 消融计划。")
    parser.add_argument("--data-root", default="数据集/ADS-B/Dataset")
    parser.add_argument("--output-prefix", default="results/stage4/adsb_target_split_conservative")
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", default="results/stage4/stage4_adsb_target_split_conservative_plan.json")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_plan(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ADS-B 保守 target split 消融计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
