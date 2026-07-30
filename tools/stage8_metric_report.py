"""汇总 Stage 8 ADS-B / LoRa 归一化代理度量 seed7 结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def weight_tag(weight_text: str) -> str:
    """保持和 Slurm 输出目录一致的权重后缀。"""
    return weight_text.strip().replace(".", "p")


def read_adsb(root: Path, weight_text: str) -> dict:
    """读取 ADS-B After R3 主方法指标。"""
    variant = f"metric_w{weight_tag(weight_text)}"
    csv_path = root / "adsb" / variant / "incremental_results.csv"
    df = pd.read_csv(csv_path)
    row = df[(df["Method"].astype(str) == "MV-ACC-CIL") & (df["Stage"].astype(str) == "After R3")].iloc[0]
    return {
        "dataset": "adsb",
        "variant": variant,
        "weight": float(weight_text),
        "cluster_count": None,
        "overall": float(row["Overall Acc"]),
        "old": float(row["Old Acc"]),
        "new": float(row["New Acc"]),
        "forgetting": float(row["Forgetting Rate"]),
        "macro_f1": float(row["Macro F1"]),
    }


def read_lora(root: Path, weight_text: str) -> dict:
    """读取 LoRa After R3 主方法指标和最终簇数。"""
    variant = f"metric_w{weight_tag(weight_text)}"
    base = root / "lora" / variant
    incremental = pd.read_csv(base / "incremental_results.csv")
    summary = pd.read_csv(base / "per_round_summary_results.csv")
    row = incremental[incremental["Stage"].astype(str) == "After R3"].iloc[0]
    r3_summary = summary[summary["Round"].astype(str) == "R3"].iloc[0]
    return {
        "dataset": "lora",
        "variant": variant,
        "weight": float(weight_text),
        "cluster_count": int(r3_summary["Cluster Count"]),
        "overall": float(row["Overall Acc"]),
        "old": float(row["Old Acc"]),
        "new": float(row["New Acc"]),
        "forgetting": float(row["Forgetting Rate"]),
        "macro_f1": float(row["Macro F1"]),
    }


def gate(row: dict, baseline: dict) -> dict:
    """按数据集预注册门槛判断是否值得扩三种子。"""
    delta_overall = row["overall"] - baseline["overall"]
    delta_old = row["old"] - baseline["old"]
    delta_new = row["new"] - baseline["new"]
    if row["dataset"] == "adsb":
        passed = (
            delta_overall >= 0.01
            and (delta_old >= 0.01 or delta_new >= 0.01)
            and delta_old >= -0.02
            and delta_new >= -0.02
        )
    else:
        passed = (
            row["cluster_count"] == 5
            and delta_overall >= 0.02
            and delta_old >= -0.03
            and delta_new >= -0.03
        )
    return {
        "dataset": row["dataset"],
        "variant": row["variant"],
        "weight": row["weight"],
        "delta_overall": delta_overall,
        "delta_old": delta_old,
        "delta_new": delta_new,
        "delta_forgetting": row["forgetting"] - baseline["forgetting"],
        "passed_seed7_gate": bool(passed),
    }


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 Stage 8 归一化代理度量 seed7 矩阵。")
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--weights", default="0,0.10,0.25")
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.matrix_root)
    weight_items = [item.strip() for item in args.weights.split(",") if item.strip()]
    rows = [read_adsb(root, item) for item in weight_items] + [read_lora(root, item) for item in weight_items]
    baselines = {name: next(row for row in rows if row["dataset"] == name and row["weight"] == 0.0) for name in ("adsb", "lora")}
    gates = [gate(row, baselines[row["dataset"]]) for row in rows if row["weight"] != 0.0]
    passing = [item for item in gates if item["passed_seed7_gate"]]

    lines = [
        "# Stage 8 归一化代理度量 seed7 报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 机制：增量训练期 cosine-proxy metric loss，只用高置信伪标签和 replay，不使用 held-out 真值。",
        "",
        "| Dataset | Variant | Clusters | Overall | Old | New | Forgetting | Macro F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['cluster_count'] if row['cluster_count'] is not None else '-'} | "
            f"{fmt(row['overall'])} | {fmt(row['old'])} | {fmt(row['new'])} | "
            f"{fmt(row['forgetting'])} | {fmt(row['macro_f1'])} |"
        )

    lines.extend([
        "",
        "## 相对各自 weight=0 的变化",
        "",
        "| Dataset | Weight | ΔOverall | ΔOld | ΔNew | ΔForgetting | 是否过 seed7 门槛 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for item in gates:
        lines.append(
            f"| {item['dataset']} | {item['weight']:.2f} | {item['delta_overall']:+.4f} | "
            f"{item['delta_old']:+.4f} | {item['delta_new']:+.4f} | "
            f"{item['delta_forgetting']:+.4f} | {'是' if item['passed_seed7_gate'] else '否'} |"
        )
    lines.append("")
    if passing:
        names = ", ".join(f"{item['dataset']}@{item['weight']:.2f}" for item in passing)
        lines.append(f"- 通过 seed7 门槛：`{names}`，下一步扩对应数据集 seed13/31。")
    else:
        lines.append("- 无候选通过 seed7 门槛；归档为负消融，不扩三种子。")
    lines.append("")

    payload = {
        "job_id": args.job_id,
        "matrix_root": args.matrix_root,
        "rows": rows,
        "gates": gates,
        "passing": passing,
        "adopt_any": bool(passing),
    }
    output = Path(args.output)
    summary = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成 Stage 8 报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
