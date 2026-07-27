"""阶段 1 CIL 后端消融矩阵生成工具。

该脚本只生成可审计的实验矩阵，不启动训练。矩阵围绕 Job 44420671
暴露的 R3 旧类遗忘风险设计，优先检查 KD、回放强度、记忆容量和
骨干解冻策略。输出 JSON 会被 Slurm 脚本和结果报告共同引用，避免
消融设置散落在 shell 命令里。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMMON_ARGS = [
    "--dataset_path 数据集/WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl",
    "--selected_rx_list 2",
    "--initial_known_classes 10",
    "--round_size 10",
    "--num_rounds 3",
    "--seed 7",
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


def build_matrix() -> list[dict[str, Any]]:
    """返回后端消融矩阵。

    每个变体只改变一个主要因素，便于从短实验结果判断 R3 遗忘来自
    蒸馏不足、回放不足、记忆容量不足，还是骨干末端联合微调导致
    表征漂移。
    """
    return [
        {
            "variant": "supcon_default_backend",
            "purpose": "复现 Job 44420671 中 SupCon 表征 + 默认 CIL 后端，作为本轮后端消融基线。",
            "args": COMMON_ARGS,
        },
        {
            "variant": "kd_off",
            "purpose": "关闭旧模型蒸馏，检查 KD 是否在当前伪标签噪声下提供有效旧类保持。",
            "args": COMMON_ARGS + ["--cil_kd_weight 0.0"],
        },
        {
            "variant": "replay_x2",
            "purpose": "加大旧类回放 CE 权重，检查 R3 遗忘是否主要来自旧类约束不足。",
            "args": COMMON_ARGS + ["--cil_replay_weight 2.0"],
        },
        {
            "variant": "memory_50",
            "purpose": "提高每类原始 IQ 回放样本数，检查记忆容量是否限制旧类保持。",
            "args": COMMON_ARGS + ["--memory_per_class 50"],
        },
        {
            "variant": "head_only",
            "purpose": "只训练扩展分类头，不联合微调骨干末端，检查 backbone drift 是否造成旧类遗忘。",
            "args": COMMON_ARGS + ["--cil_joint_epochs 0"],
        },
        {
            "variant": "low_backbone_lr",
            "purpose": "降低骨干末端学习率，保留少量适应能力同时抑制表征漂移。",
            "args": COMMON_ARGS + ["--cil_backbone_lr 1e-6"],
        },
    ]


def build_report(project_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "stage1_backend_ablation_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "basis": {
            "source_job": "44420671",
            "source_report": "results/stage1/STAGE1_WISIG_SHORT_REPORT.md",
            "diagnosis": "SupCon 改善早期遗忘但 R3 仍严重遗忘，下一步优先检查 CIL 后端。"
        },
        "matrix": build_matrix(),
        "metrics": [
            "Overall Acc",
            "Old Acc",
            "New Acc",
            "Initial Known Acc",
            "Forgetting Rate",
            "Macro F1",
            "NMI",
            "ARI",
            "Purity",
            "Hungarian Acc",
        ],
        "selection_rule": (
            "优先选择 R3 Forgetting Rate 更低且 Overall Acc 不明显下降的变体；"
            "若 head_only 明显降低遗忘，说明骨干漂移是主要风险；"
            "若 replay_x2 或 memory_50 改善明显，则优先扩展回放策略。"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成阶段 1 WiSig CIL 后端消融矩阵。")
    parser.add_argument("--project-root", default=".", help="项目根目录。")
    parser.add_argument("--output", default="results/stage1/stage1_backend_ablation_plan.json", help="输出 JSON 路径。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_report(project_root), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"阶段 1 后端消融矩阵：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
