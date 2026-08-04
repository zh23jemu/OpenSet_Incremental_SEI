#!/usr/bin/env python3
"""汇总 Stage51 LoRa 新类吸收 seed31 小矩阵结果。

Stage51 固定 Stage48 的 LoRa 正式候选配置，只改变新伪类进入分类头和
head warmup 的吸收方式。报告器只读取每个变体已经生成的 held-out
incremental_results.csv，不参与训练、选参或访问未知真值。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


STAGE48_SEED31 = {
    "overall": 0.2652380952,
    "old": 0.2250,
    "new": 0.4261904762,
    "forgetting": 0.2011904762,
}


def _r3_row(csv_path: Path) -> pd.Series:
    """读取单个变体的 R3 held-out 评估行。"""
    if not csv_path.exists():
        raise FileNotFoundError(f"missing incremental results: {csv_path}")
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"empty incremental results: {csv_path}")
    if "Stage" in frame.columns:
        r3 = frame[frame["Stage"].astype(str).str.contains("R3", na=False)]
        if not r3.empty:
            return r3.iloc[-1]
    return frame.iloc[-1]


def _metric(row: pd.Series, name: str) -> float:
    """兼容主入口输出列名，统一抽取浮点指标。"""
    aliases = {
        "overall": "Overall Acc",
        "old": "Old Acc",
        "new": "New Acc",
        "forgetting": "Forgetting Rate",
        "macro_f1": "Macro F1",
    }
    return float(row[aliases[name]])


def collect_variant(root: Path, variant: str, job_id: str) -> dict[str, float | str]:
    """收集一个 Stage51 变体的 R3 指标。"""
    save_dir = root / f"lora_s51_{variant}_{job_id}"
    row = _r3_row(save_dir / "incremental_results.csv")
    return {
        "variant": variant,
        "overall": _metric(row, "overall"),
        "old": _metric(row, "old"),
        "new": _metric(row, "new"),
        "forgetting": _metric(row, "forgetting"),
        "macro_f1": _metric(row, "macro_f1"),
    }


def add_deltas(records: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    """补充相对本 job base 和 Stage48 seed31 的差值。"""
    base = next((item for item in records if item["variant"] == "base"), records[0])
    enriched: list[dict[str, float | str]] = []
    for item in records:
        row = dict(item)
        for key in ("overall", "old", "new", "forgetting"):
            row[f"delta_base_{key}"] = float(item[key]) - float(base[key])
            row[f"delta_stage48_{key}"] = float(item[key]) - float(STAGE48_SEED31[key])
        enriched.append(row)
    return enriched


def passes_gate(item: dict[str, float | str]) -> bool:
    """预注册通过门槛：先救 New，同时不牺牲 Overall 和遗忘。"""
    if item["variant"] == "base":
        return False
    return (
        float(item["new"]) >= STAGE48_SEED31["new"] + 0.03
        and float(item["overall"]) >= STAGE48_SEED31["overall"]
        and float(item["forgetting"]) <= STAGE48_SEED31["forgetting"] + 0.05
    )


def table_md(records: list[dict[str, float | str]]) -> str:
    """生成 Markdown 表格，避免依赖 tabulate。"""
    columns = [
        "variant",
        "overall",
        "old",
        "new",
        "forgetting",
        "delta_base_overall",
        "delta_base_new",
        "delta_stage48_new",
        "gate",
    ]
    headers = [
        "Variant",
        "Overall",
        "Old",
        "New",
        "Forgetting",
        "ΔOverall vs base",
        "ΔNew vs base",
        "ΔNew vs Stage48",
        "Gate",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for item in records:
        row_values: list[str] = []
        for column in columns:
            if column == "gate":
                row_values.append("PASS" if passes_gate(item) else "FAIL")
            elif column == "variant":
                row_values.append(str(item[column]))
            else:
                row_values.append(f"{float(item[column]):.4f}")
        lines.append("| " + " | ".join(row_values) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage51 LoRa 新类吸收 seed31 报告器")
    parser.add_argument("--root", default="results/stage51")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--variants", nargs="+", default=["base", "new_proto_w0p10", "head_boost2", "imprint_scale1p5"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    records = add_deltas([collect_variant(root, variant, args.job_id) for variant in args.variants])
    passed = [item["variant"] for item in records if passes_gate(item)]
    best_new = max(records, key=lambda item: float(item["new"]))
    best_overall = max(records, key=lambda item: float(item["overall"]))

    lines = [
        f"# Stage51 LoRa 新类吸收 seed31 报告（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- 最佳 New：{best_new['variant']}，R3 New={float(best_new['new']):.4f}，Overall={float(best_new['overall']):.4f}",
        f"- 最佳 Overall：{best_overall['variant']}，R3 Overall={float(best_overall['overall']):.4f}，New={float(best_overall['new']):.4f}",
        f"- Gate: {'PASS' if passed else 'FAIL'}" + (f"，通过变体={', '.join(map(str, passed))}" if passed else "，不扩三种子"),
        "",
        "## 变体明细",
        "",
        table_md(records),
        "",
        "## 判定规则",
        "",
        "- 固定 Stage48 的 raw s28 + recording-consensus 0.65 + Chirp + joint discovery-CIL 配置，只改新类吸收机制。",
        "- 通过门槛：R3 New 至少比 Stage48 seed31 高 0.03，Overall 不低于 Stage48 seed31，Forgetting 恶化不超过 0.05。",
        "- 不通过则归档为 seed31 新类吸收负消融，不进入 seed7/13/31 正式扩展。",
        "",
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage51_lora_new_class_absorption_summary_v1",
                "job_id": args.job_id,
                "stage48_seed31": STAGE48_SEED31,
                "records": records,
                "passed_variants": passed,
                "gate": "PASS" if passed else "FAIL",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
