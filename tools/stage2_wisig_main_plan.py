"""生成阶段 2 WiSig 单种子主流程计划。

阶段 1 已通过多种子确认锁定 `ratio_2p0_replay_3p0` 作为阶段 2 主后端
候选。本脚本把该选择、运行参数、输出目录和验收指标写成 JSON 审计
产物，供 Slurm 主流程和客户汇报共同引用。
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
    # 允许从项目根目录、tools 目录或 Slurm 作业中直接运行，不要求手动设置 PYTHONPATH。
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.radcil_config import get_stage2_wisig_main_config  # noqa: E402


DEFAULT_DATASET = "数据集/WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl"


def build_plan(project_root: Path, dataset_path: str, save_dir: str, seed: int | None) -> dict[str, Any]:
    """构造阶段 2 单种子主流程审计计划。

    计划只记录“准备跑什么”和“为什么这样跑”，不混入运行后指标。实验
    结束后应由 `incremental_results.csv` 和阶段 2 报告补充真实结果。
    """
    config = get_stage2_wisig_main_config(seed)
    return {
        "schema_version": "stage2_wisig_main_single_seed_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "stage": "stage2",
        "dataset": "WiSig",
        "task": "10 known + 10 x 3 open-set class-incremental SEI",
        "strict_protocol": "Day1 70% development split into 60% train and 10% validation; 30% held-out evaluation remains isolated.",
        "locked_main_config": config.to_dict(),
        "selection_basis": {
            "source_report": "results/stage1/STAGE1_RADCIL_MULTISEED_REPORT.md",
            "source_job": "44426767",
            "decision": (
                "`ratio_2p0_replay_3p0` 与 high-replay 对照 Overall 基本持平，"
                "但旧类准确率更高、遗忘率更低，因此按 tie-break 规则进入阶段 2。"
            ),
        },
        "run": {
            "run_name": f"wisig_main_{config.variant}_seed{config.seed}",
            "dataset_path": dataset_path,
            "save_dir": save_dir,
            "experiment_entry": "experiments/exp_wisig_mvacc_cil_strict.py",
            "argv": config.experiment_args(dataset_path=dataset_path, save_dir=save_dir),
        },
        "acceptance_metrics": [
            "clustering_results.csv: MV-ACC discovery cluster count, NMI, ARI, purity, Hungarian Acc",
            "incremental_results.csv: R1/R2/R3 Overall Acc, Old Acc, New Acc, Forgetting Rate, Macro F1",
            "per_round_summary_results.csv: round-level strict protocol summary",
        ],
        "risk_controls": [
            "不读取未知轮次评估真值参与阈值、聚类或伪标签筛选。",
            "保留 IGCD-minimal 和 SimGCD-style 为 strict/minimal baseline，不作为阶段 2 主前端。",
            "本次只跑单种子完整三轮主流程，正式结论仍需后续三种子均值和标准差。",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成阶段 2 WiSig 单种子主流程计划。")
    parser.add_argument("--project-root", default=".", help="项目根目录。")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET, help="WiSig 数据集路径。")
    parser.add_argument("--save-dir", default="results/stage2/wisig_main_ratio_2p0_replay_3p0_seed7", help="实验输出目录。")
    parser.add_argument("--seed", type=int, default=7, help="单种子主流程随机种子。")
    parser.add_argument("--output", default="results/stage2/stage2_wisig_main_single_seed_plan.json", help="输出 JSON 路径。")
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
        save_dir=args.save_dir,
        seed=args.seed,
    )
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"阶段 2 WiSig 单种子主流程计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
