"""生成阶段 3 WiSig 正式三种子主实验/后端消融计划。

阶段 2 已经证明 MV-ACC + `ratio_2p0_replay_3p0` 能跑通 WiSig 完整三轮
主流程。阶段 3 的目标不是继续调参，而是在相同配置下运行 seed 7/13/31，
并为正式结果表提供均值和标准差。脚本也支持 `ratio_3p0_replay_3p0`
作为同协议 high-replay 后端消融，除旧类 batch 配比外保持主流程预算一致。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 允许 Slurm 或本地从任意工作目录直接运行，不依赖手动 PYTHONPATH。
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.radcil_config import get_radcil_wisig_config  # noqa: E402


DEFAULT_DATASET = "数据集/WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl"
DEFAULT_SEEDS = (7, 13, 31)


def build_runs(dataset_path: str, output_prefix: str, job_id: str, seeds: tuple[int, ...], variant: str) -> list[dict[str, Any]]:
    """展开正式三种子运行清单。

    每个 seed 写入独立目录，避免 checkpoint、回放记忆和 CSV 互相覆盖。
    """
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        config = get_radcil_wisig_config(variant=variant, seed=seed)
        save_dir = f"{output_prefix}_{config.variant}_seed{seed}_{job_id}"
        runs.append(
            {
                "run_name": f"wisig_main_{config.variant}_seed{seed}",
                "seed": seed,
                "variant": config.variant,
                "frontend": config.frontend,
                "save_dir": save_dir,
                "argv": config.experiment_args(dataset_path=dataset_path, save_dir=save_dir),
            }
        )
    return runs


def build_plan(project_root: Path, dataset_path: str, output_prefix: str, job_id: str, seeds: tuple[int, ...], variant: str) -> dict[str, Any]:
    """构造阶段 3 正式主实验审计计划。"""
    config = get_radcil_wisig_config(variant=variant)
    is_main = variant == "ratio_2p0_replay_3p0"
    return {
        "schema_version": "stage3_wisig_main_multiseed_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "stage": "stage3",
        "dataset": "WiSig",
        "task": "formal 3-seed 10 known + 10 x 3 open-set class-incremental SEI",
        "seeds": list(seeds),
        "selection_basis": {
            "stage2_report": "results/stage2/STAGE2_WISIG_MAIN_SINGLE_SEED_REPORT_44433946.md",
            "stage2_job": "44433946",
            "decision": (
                "阶段 2 seed7 主流程已跑通，R3 Overall 0.6328、Forgetting 0.1389；"
                "阶段 3 固定相同训练预算做正式三种子统计，不在本轮继续调参。"
            ),
            "variant_role": "main" if is_main else "formal_ablation_high_replay",
            "variant_note": (
                "`ratio_2p0_replay_3p0` 是阶段 3 主方法。"
                if is_main
                else "`ratio_3p0_replay_3p0` 是同协议 high-replay 后端消融，用于检验更高旧类 batch 配比的代价。"
            ),
        },
        "locked_config": config.to_dict(),
        "runs": build_runs(dataset_path=dataset_path, output_prefix=output_prefix, job_id=job_id, seeds=seeds, variant=variant),
        "aggregation": {
            "script": "tools/stage3_wisig_main_multiseed_report.py",
            "metrics": [
                "R1/R2/R3 Overall Acc mean/std",
                "R1/R2/R3 Old Acc mean/std",
                "R1/R2/R3 New Acc mean/std",
                "R1/R2/R3 Forgetting Rate mean/std",
                "R1/R2/R3 Macro F1 mean/std",
                "R1/R2/R3 discovery NMI/ARI/Purity/Hungarian Acc mean/std",
            ],
        },
        "risk_controls": [
            "固定阶段 2 主配置，不使用 seed 结果反向调参。",
            "正式报告必须包含 seed 7/13/31 的单种子行、均值和标准差。",
            "Seed 13 在阶段 1 已知更难，本轮特别检查其 R3 Old 与 Forgetting。",
            "high-replay 对照只改变 old:new batch ratio，不改变前端、训练轮数、数据协议或评估集。",
        ],
    }


def parse_seeds(text: str) -> tuple[int, ...]:
    """解析逗号分隔的 seed 列表，并保持顺序去重。"""
    seeds: list[int] = []
    for item in text.split(","):
        value = item.strip()
        if not value:
            continue
        seed = int(value)
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise ValueError("至少需要一个 seed。")
    return tuple(seeds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成阶段 3 WiSig 正式三种子主实验计划。")
    parser.add_argument("--project-root", default=".", help="项目根目录。")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET, help="WiSig 数据集路径。")
    parser.add_argument("--output-prefix", default="results/stage3/wisig_main", help="每个 seed 输出目录前缀。")
    parser.add_argument("--job-id", default="manual", help="Slurm Job ID，用于生成唯一输出目录。")
    parser.add_argument("--seeds", default="7,13,31", help="逗号分隔的正式实验 seed 列表。")
    parser.add_argument("--variant", default="ratio_2p0_replay_3p0", help="RADCIL WiSig 配置名。")
    parser.add_argument("--output", default="results/stage3/stage3_wisig_main_multiseed_plan.json", help="输出 JSON 路径。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    plan = build_plan(
        project_root=project_root,
        dataset_path=args.dataset_path,
        output_prefix=args.output_prefix,
        job_id=args.job_id,
        seeds=parse_seeds(args.seeds),
        variant=args.variant,
    )
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"阶段 3 WiSig 正式三种子主实验计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
