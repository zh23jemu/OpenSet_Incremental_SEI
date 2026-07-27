"""生成 replay_x2 起点的 RADCIL 后端二阶消融矩阵。

阶段 1 后端消融已经证明：
- `replay_x2` 当前最稳，说明旧类训练约束强度不足；
- `kd_off` 优于默认，说明现有 KD 目标或权重可能带来负作用；
- `head_only` 最差，说明完全冻结骨干不可行。

本脚本把这些结论转成下一轮可执行矩阵，重点比较 replay loss 权重、
旧类 batch 约束、masked KD、特征蒸馏和末端层解冻。当前项目的旧
WiSig strict 入口尚未全部支持这些新参数，因此矩阵会显式标记
`requires_code_support`，供下一步实现 RADCIL 后端模块和 Slurm 脚本时
逐项落地。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_ARGS = [
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
    "--cil_replay_weight 2.0",
]


def build_matrix() -> list[dict[str, Any]]:
    """返回下一轮 RADCIL 后端候选矩阵。"""
    return [
        {
            "variant": "replay_x2_confirm",
            "purpose": "复现阶段 1 最佳短实验，作为二阶矩阵锚点。",
            "args": BASE_ARGS,
            "requires_code_support": [],
        },
        {
            "variant": "replay_x3",
            "purpose": "继续增强旧类 replay CE，检查 replay_x2 是否尚未达到旧类约束上限。",
            "args": BASE_ARGS + ["--cil_replay_weight 3.0"],
            "requires_code_support": [],
        },
        {
            "variant": "balanced_old_new_batch",
            "purpose": "固定每个增量 batch 的旧类/新类样本比例，避免 replay loss 权重大但采样不足。",
            "args": BASE_ARGS + ["--radcil_old_new_batch_ratio 1.0"],
            "requires_code_support": ["radcil_old_new_batch_ratio"],
        },
        {
            "variant": "masked_kd_low",
            "purpose": "只对旧类 logits 做 masked KD，低权重验证 KD 是否可在去除新类干扰后恢复收益。",
            "args": BASE_ARGS + ["--cil_kd_weight 0.25", "--radcil_masked_kd"],
            "requires_code_support": ["radcil_masked_kd"],
        },
        {
            "variant": "masked_kd_schedule",
            "purpose": "旧类 masked KD 使用随轮次衰减/升温调度，避免默认 KD 在伪标签噪声下压制新类。",
            "args": BASE_ARGS + [
                "--cil_kd_weight 0.5",
                "--radcil_masked_kd",
                "--radcil_kd_schedule cosine",
                "--cil_temperature 3.0",
            ],
            "requires_code_support": ["radcil_masked_kd", "radcil_kd_schedule"],
        },
        {
            "variant": "feature_distill_tail",
            "purpose": "加入 teacher/student 末端特征蒸馏，直接约束骨干漂移。",
            "args": BASE_ARGS + ["--radcil_feature_distill_weight 0.5"],
            "requires_code_support": ["radcil_feature_distill_weight"],
        },
        {
            "variant": "tail_unfreeze_plus_feature_distill",
            "purpose": "保留末端层适应能力，同时用特征蒸馏降低旧类表征漂移。",
            "args": BASE_ARGS + [
                "--cil_backbone_lr 5e-6",
                "--radcil_feature_distill_weight 0.5",
                "--radcil_unfreeze_scope tail",
            ],
            "requires_code_support": ["radcil_feature_distill_weight", "radcil_unfreeze_scope"],
        },
    ]


def build_report(project_root: Path) -> dict[str, Any]:
    """构造 JSON 安全的矩阵报告。"""
    return {
        "schema_version": "stage1_radcil_backend_matrix_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "basis": {
            "source_report": "results/stage1/STAGE1_BACKEND_ABLATION_REPORT.md",
            "source_job": "44420869",
            "diagnosis": (
                "replay_x2 当前最佳，kd_off 优于默认，head_only 最差；"
                "下一轮应强化 replay 采样/损失约束，并重做 KD 与特征保持。"
            ),
        },
        "matrix": build_matrix(),
        "primary_selection_rule": (
            "优先选择 R3 Old Acc 更高且 Forgetting Rate 更低的变体；"
            "若 New Acc 明显下降，则降低 replay/KD 权重或改用调度。"
        ),
        "required_metrics": [
            "Overall Acc",
            "Old Acc",
            "New Acc",
            "Initial Known Acc",
            "Forgetting Rate",
            "Macro F1",
            "per-round confidence calibration if SimGCD frontend is used",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 replay_x2 起点的 RADCIL 后端二阶矩阵。")
    parser.add_argument("--project-root", default=".", help="项目根目录。")
    parser.add_argument("--output", default="results/stage1/stage1_radcil_backend_matrix.json", help="输出 JSON 路径。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_report(project_root), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"阶段 1 RADCIL 后端二阶矩阵：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
