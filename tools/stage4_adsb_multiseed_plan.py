"""生成 ADS-B 阶段 4 正式三种子计划。

seed31 已由 Job 44467424 完成，因此正式扩展只运行 seed7/13，并在报告阶段
复用 seed31 产物。所有 seed 固定使用 ADSBLongClosedSet、old:new=2.0 和
replay weight=3.0，不根据中间结果调整参数。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def experiment_args(data_root: str, save_dir: str, seed: int) -> list[str]:
    """返回与已通过 smoke test 完全一致的 strict 实验参数。"""
    return [
        "--data_root", data_root,
        "--save_dir", save_dir,
        "--initial_known_classes", "90",
        "--round_size", "10",
        "--num_rounds", "3",
        "--seed", str(seed),
        "--train_closedset",
        "--backbone", "adsb_long",
        "--epochs", "30",
        "--batch_size", "128",
        "--test_batch_size", "256",
        "--disable_cil_baselines",
        "--use_supcon",
        "--supcon_weight", "0.1",
        "--cil_head_warmup_epochs", "2",
        "--cil_joint_epochs", "8",
        "--cil_replay_weight", "3.0",
        "--radcil_old_new_batch_ratio", "2.0",
        "--disable_visualization",
    ]


def build_plan(
    project_root: Path,
    data_root: str,
    output_prefix: str,
    job_id: str,
    reused_seed31_dir: str,
) -> dict[str, Any]:
    """构造 seed7/13 新运行加 seed31 复用结果的审计计划。"""
    runs: list[dict[str, Any]] = []
    for seed in (7, 13):
        save_dir = f"{output_prefix}_seed{seed}_{job_id}"
        runs.append(
            {
                "seed": seed,
                "save_dir": save_dir,
                "source": "current_job",
                "argv": experiment_args(data_root, save_dir, seed),
            }
        )
    runs.append(
        {
            "seed": 31,
            "save_dir": reused_seed31_dir,
            "source": "reused_job_44467424",
            "argv": experiment_args(data_root, reused_seed31_dir, 31),
        }
    )
    return {
        "schema_version": "stage4_adsb_multiseed_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "stage": "stage4",
        "dataset": "ADS-B",
        "task": "formal 3-seed strict 90 known + 10 x 3 incremental SEI",
        "seeds": [7, 13, 31],
        "selection_basis": {
            "single_seed_job": "44467424",
            "decision": (
                "seed31 long RADCIL smoke test R3 Overall 0.4856，较 legacy 0.3273 提升 0.1583；"
                "固定配置扩展 seed7/13，并复用 seed31 产物。"
            ),
        },
        "locked_config": {
            "backbone": "ADSBLongClosedSet",
            "old_new_batch_ratio": 2.0,
            "replay_weight": 3.0,
            "frontend": "MV-ACC",
        },
        "runs": runs,
        "risk_controls": [
            "只运行缺失的 seed7/13，seed31 复用已验证结果。",
            "不根据 seed7/13 中间指标调整训练或发现参数。",
            "正式报告必须同时包含三种子明细、均值和样本标准差。",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 ADS-B 阶段 4 正式三种子计划。")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--data-root", default="数据集/ADS-B/Dataset")
    parser.add_argument("--output-prefix", default="results/stage4/adsb_main_long_radcil")
    parser.add_argument("--job-id", default="manual")
    parser.add_argument(
        "--reused-seed31-dir",
        default="results/stage4/adsb_main_long_radcil_seed31_44467424",
    )
    parser.add_argument("--output", default="results/stage4/stage4_adsb_multiseed_plan.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    plan = build_plan(root, args.data_root, args.output_prefix, args.job_id, args.reused_seed31_dir)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ADS-B 阶段 4 正式三种子计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
