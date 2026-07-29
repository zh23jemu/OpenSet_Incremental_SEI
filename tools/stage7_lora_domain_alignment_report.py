"""汇总 LoRa 训练期跨天特征分布对齐 seed7 二元验证结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VARIANTS = ("alignment_off", "alignment_on")


def read_variant(root: Path, variant: str) -> dict:
    """读取 IQ_7 旧类保持和 held-out R3 指标。"""
    variant_dir = root / variant
    validation = pd.read_csv(variant_dir / "iq7_validation_retention.csv")
    summary = pd.read_csv(variant_dir / "per_round_summary_results.csv")
    incremental = pd.read_csv(variant_dir / "incremental_results.csv")
    r3_validation = validation.loc[validation["Round"] == "R3"].iloc[0]
    r3_summary = summary.loc[summary["Round"] == "R3"].iloc[0]
    r3_incremental = incremental.loc[incremental["Stage"] == "After R3"].iloc[0]
    return {
        "variant": variant,
        "iq7_r3_old_acc": float(r3_validation["IQ_7 Validation Old-Class Acc"]),
        "cluster_count": int(r3_summary["Cluster Count"]),
        "overall": float(r3_incremental["Overall Acc"]),
        "old": float(r3_incremental["Old Acc"]),
        "new": float(r3_incremental["New Acc"]),
        "forgetting": float(r3_incremental["Forgetting Rate"]),
        "macro_f1": float(r3_incremental["Macro F1"]),
    }


def main() -> int:
    """按 IQ_7 选择机制，并用 held-out 做事后平衡审计。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    rows = [read_variant(Path(args.matrix_root), variant) for variant in VARIANTS]
    baseline, candidate = rows
    validation_improved = candidate["iq7_r3_old_acc"] > baseline["iq7_r3_old_acc"]
    heldout_gate = (
        candidate["overall"] >= baseline["overall"] - 0.01
        and candidate["new"] >= baseline["new"] - 0.03
        and candidate["forgetting"] < baseline["forgetting"]
    )
    adopt = bool(validation_improved and heldout_gate)

    lines = [
        "# 阶段 7 LoRa 训练期跨天特征分布对齐 seed7 报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 机制：在每轮训练中对齐当前 discovery 与历史 replay 的单位特征均值/协方差。",
        "- 选择：只看 IQ_7 R3 旧类验证准确率；IQ_8-10 仅事后审计。",
        "",
        "| Variant | IQ_7 R3 Old | Clusters | Overall | Old | New | Forgetting | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['iq7_r3_old_acc']:.4f} | {row['cluster_count']} | "
            f"{row['overall']:.4f} | {row['old']:.4f} | {row['new']:.4f} | "
            f"{row['forgetting']:.4f} | {row['macro_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            f"- IQ_7 旧类保持改善：{'通过' if validation_improved else '未通过'}。",
            f"- Held-out 平衡门槛：{'通过' if heldout_gate else '未通过'}。",
            f"- 是否采用并扩展 seed13/31：{'是' if adopt else '否'}。",
            "",
        ]
    )
    payload = {
        "job_id": args.job_id,
        "matrix_root": args.matrix_root,
        "rows": rows,
        "selection_source": "IQ_7 R3 old-class validation accuracy only",
        "validation_improved": validation_improved,
        "heldout_gate": heldout_gate,
        "adopt": adopt,
    }
    output = Path(args.output)
    summary = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
