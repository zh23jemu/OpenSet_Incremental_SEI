"""生成 balanced_old_new_batch 起点的 RADCIL ratio/weight 细化矩阵。

Job 44422380 显示 `balanced_old_new_batch` 是当前单种子最佳折中：
它比单纯 replay_x2 明显提高 R3 Overall/Old/New，同时不过度牺牲新类。
本脚本围绕该结论生成 3x3 短实验矩阵，专门比较 old:new batch ratio
与 replay loss weight 的组合。
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
]


def build_matrix() -> list[dict[str, Any]]:
    """返回 old:new ratio 与 replay weight 的 3x3 细化矩阵。"""
    rows: list[dict[str, Any]] = []
    for ratio in (1.5, 2.0, 3.0):
        for replay_weight in (2.0, 2.5, 3.0):
            variant = f"ratio_{str(ratio).replace('.', 'p')}_replay_{str(replay_weight).replace('.', 'p')}"
            rows.append(
                {
                    "variant": variant,
                    "purpose": (
                        "围绕 balanced_old_new_batch 细化旧类 batch 配比和 replay CE 权重，"
                        "检查旧类保持与新类学习的折中点。"
                    ),
                    "old_new_batch_ratio": float(ratio),
                    "cil_replay_weight": float(replay_weight),
                    "args": BASE_ARGS
                    + [
                        f"--cil_replay_weight {replay_weight}",
                        f"--radcil_old_new_batch_ratio {ratio}",
                    ],
                    "requires_code_support": [],
                }
            )
    return rows


def build_report(project_root: Path) -> dict[str, Any]:
    """构造 JSON 安全的矩阵报告。"""
    return {
        "schema_version": "stage1_radcil_ratio_weight_matrix_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "basis": {
            "source_report": "results/stage1/STAGE1_RADCIL_BACKEND_MATRIX_REPORT.md",
            "source_job": "44422380",
            "diagnosis": (
                "balanced_old_new_batch 当前单种子最佳；需要细化 old:new ratio "
                "和 replay weight，而不是优先叠加 KD 或 feature distill。"
            ),
        },
        "matrix": build_matrix(),
        "selection_rule": (
            "优先选择 R3 Overall 与 R3 Old 同时提升、R3 New 不明显低于 0.80、"
            "Forgetting 低于 replay_x2_confirm 的组合。"
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
    parser = argparse.ArgumentParser(description="生成 RADCIL old:new ratio / replay weight 细化矩阵。")
    parser.add_argument("--project-root", default=".", help="项目根目录。")
    parser.add_argument("--output", default="results/stage1/stage1_radcil_ratio_weight_matrix.json", help="输出 JSON 路径。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_report(project_root), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"阶段 1 RADCIL ratio/weight 细化矩阵：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
