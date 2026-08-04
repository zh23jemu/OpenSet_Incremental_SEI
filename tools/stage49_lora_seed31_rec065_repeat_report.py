"""汇总 Stage49 LoRa seed31 rec065 复现稳定性验证。

该报告器只读取每个 repeat 的小型 CSV 指标，不读取模型权重或 held-out
之外的任何调参信息。目的不是再次挑选参数，而是确认同一 seed、同一配置
在补齐确定性设置后是否仍存在明显端到端波动。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


STAGE48_SEED31 = {"overall": 0.2652380952, "old": 0.2250, "new": 0.4261904762, "forgetting": 0.2011904762}
STAGE47_REC065 = {"overall": 0.2985714286, "old": 0.2559523810, "new": 0.4690476190, "forgetting": 0.2440476190}


def _r3_row(csv_path: Path) -> pd.Series:
    """读取单次 repeat 的 R3 增量评估行。"""
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
    """兼容报告列名，统一返回浮点指标。"""
    aliases = {
        "overall": "Overall Acc",
        "old": "Old Acc",
        "new": "New Acc",
        "forgetting": "Forgetting Rate",
        "macro_f1": "Macro F1",
    }
    return float(row[aliases[name]])


def collect_repeat(root: Path, repeat_id: int, job_id: str) -> dict[str, float | int]:
    """收集一个 repeat 的 R3 指标。"""
    save_dir = root / f"lora_seed31_rec065_repeat{repeat_id}_{job_id}"
    row = _r3_row(save_dir / "incremental_results.csv")
    return {
        "repeat": int(repeat_id),
        "overall": _metric(row, "overall"),
        "old": _metric(row, "old"),
        "new": _metric(row, "new"),
        "forgetting": _metric(row, "forgetting"),
        "macro_f1": _metric(row, "macro_f1"),
    }


def summarize(records: list[dict[str, float | int]], key: str) -> dict[str, float]:
    """计算均值、标准差、最小值、最大值和极差。"""
    series = pd.Series([float(item[key]) for item in records], dtype="float64")
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "max": float(series.max()),
        "range": float(series.max() - series.min()),
    }


def fmt_stats(stats: dict[str, float]) -> str:
    """格式化均值和波动范围。"""
    return f"{stats['mean']:.4f}±{stats['std']:.4f}，range={stats['range']:.4f}"


def records_to_markdown(records: list[dict[str, float | int]]) -> str:
    """生成不依赖 tabulate 的 Markdown 表格，保证远端最小环境也能运行。"""
    columns = ["repeat", "overall", "old", "new", "forgetting", "macro_f1"]
    headers = ["Repeat", "Overall", "Old", "New", "Forgetting", "Macro F1"]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for item in records:
        values: list[str] = []
        for column in columns:
            value = item[column]
            if isinstance(value, int):
                values.append(str(value))
            else:
                values.append(f"{float(value):.4f}")
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage49 LoRa seed31 rec065 repeat 报告器")
    parser.add_argument("--root", default="results/stage49")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--repeats", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    records = [collect_repeat(root, repeat_id, args.job_id) for repeat_id in args.repeats]
    summary = {key: summarize(records, key) for key in ("overall", "old", "new", "forgetting", "macro_f1")}

    mean = {key: summary[key]["mean"] for key in ("overall", "old", "new", "forgetting")}
    delta48 = {key: mean[key] - STAGE48_SEED31[key] for key in mean}
    delta47 = {key: mean[key] - STAGE47_REC065[key] for key in mean}

    # 门槛只用于解释复现稳定性：同一配置三次极差越小，越能把差异归因给算法而非训练波动。
    stable = summary["overall"]["range"] <= 0.01 and summary["new"]["range"] <= 0.03
    improves_stage48 = delta48["overall"] >= 0.01 and delta48["new"] >= 0.03 and delta48["forgetting"] <= 0.05
    gate = "PASS" if stable and improves_stage48 else "FAIL"

    table_md = records_to_markdown(records)
    lines = [
        f"# Stage49 LoRa Seed31 rec065 repeat 报告（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- Repeat 均值 Overall/Old/New/Forgetting = {summary['overall']['mean']:.4f}/{summary['old']['mean']:.4f}/{summary['new']['mean']:.4f}/{summary['forgetting']['mean']:.4f}",
        f"- Repeat 波动：Overall {fmt_stats(summary['overall'])}；New {fmt_stats(summary['new'])}",
        f"- vs Stage48 seed31：Overall {delta48['overall']:+.4f}，Old {delta48['old']:+.4f}，New {delta48['new']:+.4f}，Forgetting {delta48['forgetting']:+.4f}",
        f"- vs Stage47 rec065：Overall {delta47['overall']:+.4f}，Old {delta47['old']:+.4f}，New {delta47['new']:+.4f}，Forgetting {delta47['forgetting']:+.4f}",
        f"- Gate: {gate}",
        "",
        "## Repeat 明细",
        "",
        table_md,
        "",
        "## 判定规则",
        "",
        "- 如果同一 seed31 三次 Overall range <= 0.01 且 New range <= 0.03，认为确定性修复后复现稳定。",
        "- 如果稳定且相对 Stage48 seed31 的 Overall >= +0.01、New >= +0.03、Forgetting 不恶化超过 +0.05，才进入三种子重跑。",
        "- 否则优先归档为 LoRa 训练波动/吸收风险，不用单次高值替换正式结果。",
        "",
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage49_lora_seed31_rec065_repeat_summary_v1",
                "job_id": args.job_id,
                "records": records,
                "summary": summary,
                "stage48_seed31": STAGE48_SEED31,
                "stage47_rec065": STAGE47_REC065,
                "delta_vs_stage48_seed31": delta48,
                "delta_vs_stage47_rec065": delta47,
                "stable": stable,
                "improves_stage48": improves_stage48,
                "gate": gate,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
