"""汇总 LoRa 增量双视图一致性 seed7 矩阵并执行预注册判定。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


WEIGHTS = (0.0, 0.05, 0.10)


def variant_name(weight: float) -> str:
    """将候选权重转换为稳定目录名。"""
    return f"consistency_{weight:.2f}".replace(".", "p")


def read_variant(root: Path, weight: float) -> dict:
    """读取一个候选的 IQ_7 选择指标和 held-out R3 事后指标。"""
    variant_dir = root / variant_name(weight)
    validation = pd.read_csv(variant_dir / "iq7_validation_retention.csv")
    summary = pd.read_csv(variant_dir / "per_round_summary_results.csv")
    incremental = pd.read_csv(variant_dir / "incremental_results.csv")
    r3_validation = validation.loc[validation["Round"] == "R3"].iloc[0]
    r3_summary = summary.loc[summary["Round"] == "R3"].iloc[0]
    r3_incremental = incremental.loc[incremental["Stage"] == "After R3"].iloc[0]
    return {
        "weight": float(weight),
        "variant": variant_name(weight),
        "iq7_r3_old_acc": float(r3_validation["IQ_7 Validation Old-Class Acc"]),
        "cluster_count": int(r3_summary["Cluster Count"]),
        "overall": float(r3_incremental["Overall Acc"]),
        "old": float(r3_incremental["Old Acc"]),
        "new": float(r3_incremental["New Acc"]),
        "forgetting": float(r3_incremental["Forgetting Rate"]),
        "macro_f1": float(r3_incremental["Macro F1"]),
    }


def main() -> int:
    """按 IQ_7 选择候选，并用 held-out 指标执行事后风险门槛。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    rows = [read_variant(Path(args.matrix_root), weight) for weight in WEIGHTS]
    baseline = next(row for row in rows if row["weight"] == 0.0)
    selected = sorted(
        rows, key=lambda row: (-row["iq7_r3_old_acc"], row["weight"])
    )[0]
    validation_improved = selected["iq7_r3_old_acc"] > baseline["iq7_r3_old_acc"]
    heldout_gate = (
        selected["overall"] >= baseline["overall"] - 0.01
        and selected["new"] >= baseline["new"] - 0.03
        and selected["forgetting"] < baseline["forgetting"]
    )
    expand_multiseed = bool(validation_improved and heldout_gate)

    lines = [
        "# 阶段 6 LoRa 增量双视图一致性 seed7 报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 候选选择：只按 IQ_7 R3 初始旧类验证准确率；并列选择较小权重。",
        "- IQ_8-10 指标：候选固定后才做事后审计，不参与选择。",
        "",
        "| Consistency Weight | IQ_7 R3 Old | Clusters | Overall | Old | New | Forgetting | Macro F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['weight']:.2f} | {row['iq7_r3_old_acc']:.4f} | "
            f"{row['cluster_count']} | {row['overall']:.4f} | {row['old']:.4f} | "
            f"{row['new']:.4f} | {row['forgetting']:.4f} | {row['macro_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            f"- IQ_7 选择权重：`{selected['weight']:.2f}`。",
            f"- IQ_7 旧类保持改善：{'通过' if validation_improved else '未通过'}。",
            f"- Held-out 平衡门槛：{'通过' if heldout_gate else '未通过'}。",
            f"- 是否扩展 seed13/31：{'是' if expand_multiseed else '否'}。",
            "",
        ]
    )

    payload = {
        "job_id": args.job_id,
        "matrix_root": args.matrix_root,
        "selection_source": "IQ_7 R3 old-class validation accuracy only",
        "rows": rows,
        "selected": selected,
        "validation_improved": validation_improved,
        "heldout_gate": heldout_gate,
        "expand_multiseed": expand_multiseed,
    }
    output = Path(args.output)
    summary = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已生成报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
