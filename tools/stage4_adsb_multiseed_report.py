"""汇总 ADS-B 阶段 4 正式三种子结果并生成 Markdown 报告。"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROUNDS = ("R1", "R2", "R3")
METRICS = ("overall", "old", "new", "forgetting", "macro_f1", "nmi", "ari", "hungarian")


def read_rows(path: Path) -> list[dict[str, str]]:
    """读取标准结果 CSV；缺失文件时失败，避免生成不完整正式报告。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少正式结果文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def by_round(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """兼容 Round=R1 与 Stage=After R1 两种字段。"""
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        text = f"{row.get('Round', '')} {row.get('Stage', '')}"
        for round_name in ROUNDS:
            if round_name in text:
                result[round_name] = row
    return result


def number(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def extract(save_dir: Path, seed: int) -> list[dict[str, Any]]:
    """提取单个 seed 的发现与增量指标。"""
    clusters = by_round(read_rows(save_dir / "clustering_results.csv"))
    increments = by_round(read_rows(save_dir / "incremental_results.csv"))
    result: list[dict[str, Any]] = []
    for round_name in ROUNDS:
        cluster = clusters[round_name]
        inc = increments[round_name]
        result.append(
            {
                "seed": seed,
                "round": round_name,
                "cluster_count": int(float(cluster["Final Cluster Count"])),
                "overall": number(inc.get("Overall Acc")),
                "old": number(inc.get("Old Acc")),
                "new": number(inc.get("New Acc")),
                "forgetting": number(inc.get("Forgetting Rate")),
                "macro_f1": number(inc.get("Macro F1")),
                "nmi": number(cluster.get("NMI")),
                "ari": number(cluster.get("ARI")),
                "hungarian": number(cluster.get("Hungarian Acc")),
            }
        )
    return result


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按轮次计算三种子均值和样本标准差。"""
    summary: list[dict[str, Any]] = []
    for round_name in ROUNDS:
        selected = [row for row in rows if row["round"] == round_name]
        item: dict[str, Any] = {"round": round_name, "seed_count": len(selected)}
        for metric in METRICS:
            values = [float(row[metric]) for row in selected if row[metric] is not None]
            item[f"{metric}_mean"] = statistics.mean(values)
            item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(item)
    return summary


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def build_report(plan: dict[str, Any], rows: list[dict[str, Any]], summary: list[dict[str, Any]], job_id: str) -> str:
    """生成正式三种子报告，并明确当前剩余的欠聚类风险。"""
    lines = [
        "# 阶段 4 ADS-B Long-RADCIL 正式三种子报告",
        "",
        f"- 生成时间 UTC：{datetime.now(timezone.utc).isoformat()}",
        f"- Slurm Job：`{job_id}`（seed7/13）；seed31 复用 Job `44467424`。",
        "- 协议：ADS-B strict 90+10x3，ADSBLongClosedSet，old:new=2.0，replay weight=3.0。",
        "",
        "## R3 单种子明细",
        "",
        "| Seed | Clusters | Overall | Old | New | Forgetting | Macro F1 | NMI | ARI | Hungarian Acc |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [item for item in rows if item["round"] == "R3"]:
        lines.append(
            f"| {row['seed']} | {row['cluster_count']} | {fmt(row['overall'])} | {fmt(row['old'])} | "
            f"{fmt(row['new'])} | {fmt(row['forgetting'])} | {fmt(row['macro_f1'])} | "
            f"{fmt(row['nmi'])} | {fmt(row['ari'])} | {fmt(row['hungarian'])} |"
        )
    lines.extend([
        "",
        "## 三轮均值与标准差",
        "",
        "| 轮次 | Overall | Old | New | Forgetting | Macro F1 | NMI | ARI | Hungarian Acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in summary:
        lines.append(
            f"| {item['round']} | {fmt(item['overall_mean'])}±{fmt(item['overall_std'])} | "
            f"{fmt(item['old_mean'])}±{fmt(item['old_std'])} | {fmt(item['new_mean'])}±{fmt(item['new_std'])} | "
            f"{fmt(item['forgetting_mean'])}±{fmt(item['forgetting_std'])} | "
            f"{fmt(item['macro_f1_mean'])}±{fmt(item['macro_f1_std'])} | "
            f"{fmt(item['nmi_mean'])}±{fmt(item['nmi_std'])} | {fmt(item['ari_mean'])}±{fmt(item['ari_std'])} | "
            f"{fmt(item['hungarian_mean'])}±{fmt(item['hungarian_std'])} |"
        )
    r3 = next(item for item in summary if item["round"] == "R3")
    lines.extend([
        "",
        "## 当前判断",
        "",
        f"- R3 Overall：`{fmt(r3['overall_mean'])}±{fmt(r3['overall_std'])}`。",
        f"- R3 Old：`{fmt(r3['old_mean'])}`，New：`{fmt(r3['new_mean'])}`，Forgetting：`{fmt(r3['forgetting_mean'])}`。",
        "- long backbone 与 RADCIL 配比迁移已完成；若各 seed 仍系统性少于 10 簇，下一步聚焦发现前端欠聚类，而不是继续修改后端。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 ADS-B 阶段 4 正式三种子结果。")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for run in plan["runs"]:
        rows.extend(extract(Path(run["save_dir"]), int(run["seed"])))
    summary = summarize(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(plan, rows, summary, args.job_id), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps({"plan": plan, "per_seed_rounds": rows, "summary": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"ADS-B 阶段 4 正式三种子报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
