"""生成 RADCIL 两个候选后端的多种子确认计划。

阶段 1 ratio/weight 细化矩阵已经把完整 3x3 网格收敛到两个候选：

1. `ratio_3p0_replay_3p0`：单种子 R3 Overall 最优。
2. `ratio_2p0_replay_3p0`：单种子 R3 Forgetting 最低，Overall 与最优几乎持平。

本脚本只生成这两个候选在多个随机种子下的确认计划，避免继续扩大搜索空间。
输出 JSON 既给 Slurm 脚本留作审计产物，也方便后续报告明确每个运行目录、
种子、old:new batch ratio 与 replay weight 的对应关系。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEEDS = (7, 13, 31)

CANDIDATES = (
    {
        "variant": "ratio_3p0_replay_3p0",
        "old_new_batch_ratio": 3.0,
        "cil_replay_weight": 3.0,
        "reason": "单种子 R3 Overall 最优，适合验证总体准确率收益是否稳定。",
    },
    {
        "variant": "ratio_2p0_replay_3p0",
        "old_new_batch_ratio": 2.0,
        "cil_replay_weight": 3.0,
        "reason": "单种子 R3 Forgetting 最低且 Overall 几乎持平，适合验证旧类保持收益是否稳定。",
    },
)

BASE_ARGS = [
    "--dataset_path 数据集/WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl",
    "--selected_rx_list 2",
    "--initial_known_classes 10",
    "--round_size 10",
    "--num_rounds 3",
    "--train_closedset",
    "--epochs 8",
    "--batch_size 128",
    "--test_batch_size 256",
    "--disable_visualization",
    "--disable_cil_baselines",
    "--use_supcon",
    "--supcon_weight 0.1",
    "--cil_head_warmup_epochs 1",
    "--cil_joint_epochs 3",
]


def build_runs() -> list[dict[str, Any]]:
    """展开两个候选配置和三个种子，返回可审计的运行清单。"""
    runs: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for seed in SEEDS:
            run_name = f"{candidate['variant']}_seed{seed}"
            runs.append(
                {
                    "run_name": run_name,
                    "variant": candidate["variant"],
                    "seed": int(seed),
                    "old_new_batch_ratio": float(candidate["old_new_batch_ratio"]),
                    "cil_replay_weight": float(candidate["cil_replay_weight"]),
                    "selection_reason": candidate["reason"],
                    "args": BASE_ARGS
                    + [
                        f"--seed {seed}",
                        f"--cil_replay_weight {candidate['cil_replay_weight']}",
                        f"--radcil_old_new_batch_ratio {candidate['old_new_batch_ratio']}",
                    ],
                    "expected_output_dir_pattern": f"results/stage1/wisig_radcil_multiseed_{run_name}_<jobid>",
                }
            )
    return runs


def build_report(project_root: Path) -> dict[str, Any]:
    """构造多种子确认计划报告。

    报告只描述候选、种子和选择规则，不包含实验结果；结果由 Slurm 运行后
    `incremental_results.csv`、`per_round_summary_results.csv` 和后续汇总报告记录。
    """
    return {
        "schema_version": "stage1_radcil_multiseed_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "basis": {
            "source_report": "results/stage1/STAGE1_RADCIL_RATIO_WEIGHT_REPORT.md",
            "source_job": "44422703",
            "diagnosis": (
                "ratio_3p0_replay_3p0 单种子 R3 Overall 最优；"
                "ratio_2p0_replay_3p0 单种子遗忘率最低且 Overall 几乎持平。"
            ),
        },
        "seeds": list(SEEDS),
        "candidates": list(CANDIDATES),
        "runs": build_runs(),
        "selection_rule": (
            "优先比较 3 种子 R3 Overall、Old Acc、New Acc、Forgetting Rate 和 Macro F1 的均值与标准差；"
            "若 Overall 差异小于 0.01，则优先选择 Forgetting 更低且 Old Acc 更高的候选。"
        ),
        "required_metrics": [
            "Overall Acc",
            "Old Acc",
            "New Acc",
            "Initial Known Acc",
            "Forgetting Rate",
            "Macro F1",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 RADCIL 多种子确认计划。")
    parser.add_argument("--project-root", default=".", help="项目根目录。")
    parser.add_argument("--output", default="results/stage1/stage1_radcil_multiseed_plan.json", help="输出 JSON 路径。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_report(project_root), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"阶段 1 RADCIL 多种子确认计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
