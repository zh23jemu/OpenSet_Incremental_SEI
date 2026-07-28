"""汇总 LoRa 训练期旧类原型锚定 seed7 预注册矩阵。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


WEIGHTS = (0.0, 0.25, 1.0)


def variant_name(weight: float) -> str:
    """将预注册权重转换为稳定目录名。"""
    return f"anchor_{str(weight).replace('.', 'p')}"


def read_variant(root: Path, weight: float) -> dict:
    """读取一个变体的 IQ_7 选择指标和严格 held-out 事后指标。"""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    rows = [read_variant(Path(args.matrix_root), weight) for weight in WEIGHTS]
    # 正式候选只按 IQ_7 R3 旧类保持率选择；并列时取较小权重。下方 held-out
    # 指标在选择完成后才用于事后审计，不能反向改变所选候选。
    selected = sorted(rows, key=lambda row: (-row["iq7_r3_old_acc"], row["weight"]))[0]
    baseline = next(row for row in rows if row["weight"] == 0.0)
    iq7_improved = selected["iq7_r3_old_acc"] > baseline["iq7_r3_old_acc"]
    heldout_tradeoff_improved = (
        selected["old"] > baseline["old"]
        and selected["new"] >= baseline["new"] - 0.03
        and selected["forgetting"] < baseline["forgetting"]
    )
    expand_multiseed = bool(selected["weight"] > 0 and iq7_improved and heldout_tradeoff_improved)

    lines = [
        "# 阶段 5 LoRa 训练期旧类原型锚定 seed7 报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 候选选择：只按 IQ_7 的 R3 初始旧类验证准确率；并列选择较小权重。",
        "- IQ_8-10 指标：仅在候选固定后做事后审计，不参与选择。",
        "",
        "| Anchor Weight | IQ_7 R3 Old | Clusters | Overall | Old | New | Forgetting | Macro F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['weight']:.2f} | {row['iq7_r3_old_acc']:.4f} | {row['cluster_count']} | "
            f"{row['overall']:.4f} | {row['old']:.4f} | {row['new']:.4f} | "
            f"{row['forgetting']:.4f} | {row['macro_f1']:.4f} |"
        )
    lines.extend([
        "",
        "## 判定",
        "",
        f"- IQ_7 选择权重：`{selected['weight']:.2f}`。",
        f"- IQ_7 旧类保持改善：{'通过' if iq7_improved else '未通过'}。",
        f"- Held-out 新旧类权衡事后门槛：{'通过' if heldout_tradeoff_improved else '未通过'}。",
        f"- 是否扩展 seed13/31：{'是' if expand_multiseed else '否'}。",
        "",
    ])
    payload = {
        "job_id": args.job_id,
        "matrix_root": args.matrix_root,
        "selection_source": "IQ_7 R3 old-class validation accuracy only",
        "rows": rows,
        "selected": selected,
        "iq7_improved": iq7_improved,
        "heldout_tradeoff_improved": heldout_tradeoff_improved,
        "expand_multiseed": expand_multiseed,
    }
    output = Path(args.output)
    summary = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"LoRa 原型锚定报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
