"""汇总 ADS-B target split + 训练期旧类原型锚定 seed7 消融结果。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def weight_tag(weight_text: str) -> str:
    """保持和计划脚本、Slurm 循环一致的权重目录后缀。"""

    return weight_text.strip().replace(".", "p")


def load_r3_row(output_prefix: str, job_id: str, weight_text: str) -> dict[str, Any]:
    """读取某个权重的 After R3 主方法指标。"""

    weight = float(weight_text)
    save_dir = Path(f"{output_prefix}_w{weight_tag(weight_text)}_{job_id}")
    csv_path = save_dir / "incremental_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"缺少增量结果：{csv_path}")
    df = pd.read_csv(csv_path)
    row = df[(df["Method"].astype(str) == "MV-ACC-CIL") & (df["Stage"].astype(str) == "After R3")]
    if row.empty:
        raise RuntimeError(f"{csv_path} 中缺少 MV-ACC-CIL After R3")
    item = row.iloc[0].to_dict()
    return {
        "anchor_weight": weight,
        "save_dir": str(save_dir),
        "overall": float(item["Overall Acc"]),
        "old": float(item["Old Acc"]),
        "new": float(item["New Acc"]),
        "forgetting": float(item["Forgetting Rate"]),
        "macro_f1": float(item["Macro F1"]),
    }


def evaluate_candidate(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """按预注册 seed7 门槛判断是否值得扩三种子。"""

    delta_overall = float(row["overall"]) - float(baseline["overall"])
    delta_old = float(row["old"]) - float(baseline["old"])
    delta_new = float(row["new"]) - float(baseline["new"])
    delta_forgetting = float(row["forgetting"]) - float(baseline["forgetting"])
    passed = (
        delta_overall >= -0.002
        and (delta_old >= 0.010 or delta_forgetting <= -0.010)
        and delta_new >= -0.020
    )
    return {
        "anchor_weight": row["anchor_weight"],
        "delta_overall": delta_overall,
        "delta_old": delta_old,
        "delta_new": delta_new,
        "delta_forgetting": delta_forgetting,
        "passed_seed7_gate": bool(passed),
    }


def fmt(value: float) -> str:
    """四位小数格式化。"""

    return f"{value:.4f}"


def fmt_delta(value: float) -> str:
    """带符号四位小数格式化。"""

    return f"{value:+.4f}"


def build_report(job_id: str, output_prefix: str, rows: list[dict[str, Any]], gates: list[dict[str, Any]]) -> str:
    """生成 Markdown 报告，直接回答是否继续扩三种子。"""

    baseline = next(item for item in rows if float(item["anchor_weight"]) == 0.0)
    passing = [item for item in gates if item["passed_seed7_gate"]]
    lines = [
        "# 阶段 4 ADS-B Target Split 原型锚定 seed7 消融",
        "",
        f"- 生成时间 UTC：{datetime.now(timezone.utc).isoformat()}",
        f"- Slurm Job：`{job_id}`；seed：`7`。",
        f"- 输出前缀：`{output_prefix}`",
        "- 固定前端：默认 target split `clusters=10/max_added=4/silhouette=0.26`。",
        "- 变量：`radcil_prototype_anchor_weight`，其余 ADS-B Long-RADCIL 后端参数保持不变。",
        "",
        "## R3 指标",
        "",
        "| Anchor Weight | Overall | Old | New | Forgetting | Macro F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['anchor_weight']:.2f} | {fmt(row['overall'])} | {fmt(row['old'])} | "
            f"{fmt(row['new'])} | {fmt(row['forgetting'])} | {fmt(row['macro_f1'])} |"
        )

    lines.extend([
        "",
        "## 相对 weight=0 的变化",
        "",
        "| Anchor Weight | ΔOverall | ΔOld | ΔNew | ΔForgetting | 是否过 seed7 门槛 |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for gate in gates:
        if float(gate["anchor_weight"]) == 0.0:
            continue
        lines.append(
            f"| {gate['anchor_weight']:.2f} | {fmt_delta(gate['delta_overall'])} | "
            f"{fmt_delta(gate['delta_old'])} | {fmt_delta(gate['delta_new'])} | "
            f"{fmt_delta(gate['delta_forgetting'])} | {'是' if gate['passed_seed7_gate'] else '否'} |"
        )

    lines.extend([
        "",
        "## 判定",
        "",
        "- seed7 扩展门槛：Overall 不低于 baseline 0.002，且 Old 提升至少 0.010 或 Forgetting 降低至少 0.010，同时 New 下降不超过 0.020。",
    ])
    if passing:
        weights = ", ".join(f"{item['anchor_weight']:.2f}" for item in passing)
        lines.append(f"- 通过门槛的权重：`{weights}`。下一步可扩展 seed13/31 做三种子确认。")
    else:
        lines.append("- 没有非零权重通过 seed7 门槛；训练期旧类原型锚定应归档为负消融，不扩三种子。")
    lines.append("- 本报告只使用 held-out 指标做事后审计；候选机制已在计划中预注册，不能用 R3 真值反复调权重。")
    lines.append("")
    lines.append(f"- weight=0 baseline R3：Overall `{fmt(baseline['overall'])}`，Old `{fmt(baseline['old'])}`，New `{fmt(baseline['new'])}`，Forgetting `{fmt(baseline['forgetting'])}`。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 ADS-B target split 原型锚定 seed7 消融。")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--weights", default="0,0.10,0.25")
    parser.add_argument("--output-prefix", default="results/stage4/adsb_target_split_anchor_seed7")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    weight_items = [item.strip() for item in args.weights.split(",") if item.strip()]
    weights = [float(item) for item in weight_items]
    rows = [load_r3_row(args.output_prefix, args.job_id, item) for item in weight_items]
    baseline = next(item for item in rows if float(item["anchor_weight"]) == 0.0)
    gates = [
        {
            "anchor_weight": 0.0,
            "delta_overall": 0.0,
            "delta_old": 0.0,
            "delta_new": 0.0,
            "delta_forgetting": 0.0,
            "passed_seed7_gate": False,
        }
    ] + [evaluate_candidate(row, baseline) for row in rows if float(row["anchor_weight"]) != 0.0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(args.job_id, args.output_prefix, rows, gates), encoding="utf-8")

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "job_id": args.job_id,
                "seed": 7,
                "weights": weights,
                "weight_items": weight_items,
                "output_prefix": args.output_prefix,
                "r3_rows": rows,
                "seed7_gates": gates,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"ADS-B target split 原型锚定 seed7 报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
