"""汇总阶段 3 WiSig 正式三种子主实验/后端消融结果。

输入为阶段 3 计划 JSON。脚本读取每个 seed 输出目录中的三份标准 CSV，
生成单种子明细、R1/R2/R3 均值和标准差，以及客户汇报可直接引用的
Markdown 报告。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROUND_ORDER = ("R1", "R2", "R3")
METRIC_KEYS = (
    "cluster_count",
    "nmi",
    "ari",
    "purity",
    "hungarian_acc",
    "overall_acc",
    "old_acc",
    "new_acc",
    "forgetting_rate",
    "macro_f1",
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """读取 CSV；正式报告缺任一文件都应失败，避免产生不完整结论。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少结果文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def detect_round(row: dict[str, str]) -> str | None:
    """兼容 `Round=R1` 和 `Stage=After R1` 两类结果字段。"""
    text = " ".join([row.get("Round", ""), row.get("Stage", "")])
    for round_name in ROUND_ORDER:
        if round_name in text:
            return round_name
    return None


def rows_by_round(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """按 R1/R2/R3 建立索引。"""
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        round_name = detect_round(row)
        if round_name:
            indexed[round_name] = row
    return indexed


def to_float(value: str | None) -> float | None:
    """将 CSV 单元格转换为浮点数；空值保留为 None。"""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value: Any) -> str:
    """报告表格的统一数值格式。"""
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.4f}"
    return str(value)


def extract_run_metrics(save_dir: Path, seed: int, variant: str) -> list[dict[str, Any]]:
    """提取单个 seed 的 R1/R2/R3 指标。"""
    clustering = rows_by_round(read_csv_rows(save_dir / "clustering_results.csv"))
    incremental = rows_by_round(read_csv_rows(save_dir / "incremental_results.csv"))
    summary = rows_by_round(read_csv_rows(save_dir / "per_round_summary_results.csv"))

    rows: list[dict[str, Any]] = []
    for round_name in ROUND_ORDER:
        cluster_row = clustering.get(round_name, {})
        inc_row = incremental.get(round_name, {})
        summary_row = summary.get(round_name, {})
        rows.append(
            {
                "seed": seed,
                "variant": variant,
                "round": round_name,
                "save_dir": str(save_dir),
                "cluster_count": to_float(summary_row.get("Final Cluster Count") or cluster_row.get("Final Cluster Count")),
                "nmi": to_float(summary_row.get("NMI") or cluster_row.get("NMI")),
                "ari": to_float(summary_row.get("ARI") or cluster_row.get("ARI")),
                "purity": to_float(summary_row.get("Purity") or cluster_row.get("Purity")),
                "hungarian_acc": to_float(summary_row.get("Hungarian Acc") or cluster_row.get("Hungarian Acc")),
                "overall_acc": to_float(summary_row.get("Overall Acc") or inc_row.get("Overall Acc")),
                "old_acc": to_float(inc_row.get("Old Acc")),
                "new_acc": to_float(summary_row.get("New Acc") or inc_row.get("New Acc")),
                "forgetting_rate": to_float(summary_row.get("Forgetting Rate") or inc_row.get("Forgetting Rate")),
                "macro_f1": to_float(inc_row.get("Macro F1")),
            }
        )
    return rows


def mean_std(values: list[float | None]) -> dict[str, float | None]:
    """计算均值和样本标准差；有效样本不足 2 个时标准差为 0 或 NA。"""
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return {"mean": None, "std": None}
    if len(clean) == 1:
        return {"mean": clean[0], "std": 0.0}
    return {"mean": statistics.mean(clean), "std": statistics.stdev(clean)}


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按轮次聚合所有 seed 的均值和标准差。"""
    summary: list[dict[str, Any]] = []
    for round_name in ROUND_ORDER:
        round_rows = [row for row in rows if row["round"] == round_name]
        item: dict[str, Any] = {"round": round_name, "seed_count": len(round_rows)}
        for metric in METRIC_KEYS:
            stats = mean_std([row.get(metric) for row in round_rows])
            item[f"{metric}_mean"] = stats["mean"]
            item[f"{metric}_std"] = stats["std"]
        summary.append(item)
    return summary


def build_report(plan: dict[str, Any], rows: list[dict[str, Any]], summary: list[dict[str, Any]], job_id: str) -> str:
    """生成正式三种子 Markdown 报告。"""
    locked = plan.get("locked_config", {})
    role = plan.get("selection_basis", {}).get("variant_role", "main")
    variant = locked.get("variant", "ratio_2p0_replay_3p0")
    if role == "main":
        title = "阶段 3 WiSig RADCIL 正式三种子报告"
        conclusion = "- 该报告是 WiSig 正式三种子主结果，可进入客户进度汇报。"
    elif role == "formal_hybrid_doi":
        title = "阶段 3 WiSig Hybrid RADCIL + DOI-memory 正式三种子报告"
        conclusion = "- 该报告用于判断 DOI-memory late fusion 是否稳定解决共享发现后端上限风险。"
    else:
        title = "阶段 3 WiSig RADCIL high-replay 正式消融报告"
        conclusion = "- 该报告是 WiSig 正式同协议 high-replay 后端消融，用于比较旧类保持、新类吸收与遗忘代价。"
    lines = [
        f"# {title}",
        "",
        f"- 生成时间 UTC：{datetime.now(timezone.utc).isoformat()}",
        f"- Slurm Job：`{job_id}`",
        f"- 主前端：`{locked.get('frontend', 'MV-ACC')}`",
        f"- 后端配置：`{variant}`",
        f"- Seeds：`{', '.join(str(seed) for seed in plan.get('seeds', []))}`",
        "- 协议：WiSig strict 10+10x3，固定阶段 2 训练预算，不根据 seed 结果反向调参。",
        "",
        "## R3 单种子明细",
        "",
        "| Seed | Overall | Old | New | Forgetting | Macro F1 | NMI | ARI | Hungarian Acc |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in [item for item in rows if item["round"] == "R3"]:
        lines.append(
            "| {seed} | {overall_acc} | {old_acc} | {new_acc} | {forgetting_rate} | {macro_f1} | {nmi} | {ari} | {hungarian_acc} |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )

    lines.extend(
        [
            "",
            "## 三轮均值与标准差",
            "",
            "| 轮次 | Overall 均值 | Overall 标准差 | Old 均值 | Old 标准差 | New 均值 | New 标准差 | Forgetting 均值 | Forgetting 标准差 | Macro F1 均值 | Macro F1 标准差 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary:
        lines.append(
            "| {round} | {overall_acc_mean} | {overall_acc_std} | {old_acc_mean} | {old_acc_std} | {new_acc_mean} | {new_acc_std} | {forgetting_rate_mean} | {forgetting_rate_std} | {macro_f1_mean} | {macro_f1_std} |".format(
                **{key: fmt(value) for key, value in item.items()}
            )
        )

    r3 = next((item for item in summary if item["round"] == "R3"), {})
    lines.extend(
        [
            "",
            "## 当前判断",
            "",
            f"- R3 Overall 均值：`{fmt(r3.get('overall_acc_mean'))}`，标准差：`{fmt(r3.get('overall_acc_std'))}`。",
            f"- R3 Old 均值：`{fmt(r3.get('old_acc_mean'))}`，Forgetting 均值：`{fmt(r3.get('forgetting_rate_mean'))}`。",
            conclusion,
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总阶段 3 WiSig 正式三种子主实验结果。")
    parser.add_argument("--plan", required=True, help="阶段 3 计划 JSON。")
    parser.add_argument("--job-id", default="manual", help="Slurm Job ID。")
    parser.add_argument("--output", required=True, help="Markdown 报告输出路径。")
    parser.add_argument("--summary-json", required=True, help="JSON 指标摘要输出路径。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for run in plan["runs"]:
        rows.extend(extract_run_metrics(Path(run["save_dir"]), seed=int(run["seed"]), variant=str(run["variant"])))

    summary = aggregate(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(plan, rows, summary, args.job_id), encoding="utf-8")

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"plan": plan, "per_seed_rounds": rows, "summary": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"阶段 3 WiSig 三种子报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
