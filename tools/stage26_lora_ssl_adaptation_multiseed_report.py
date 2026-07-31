"""汇总 Stage 26 LoRa SSL + cross-day 三种子结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    """读取 CSV 结果文件。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"结果为空：{path}")
    return rows


def _r3(path: Path) -> dict[str, str]:
    """取 After R3 行作为最终对比指标。"""

    rows = _read_rows(path)
    for row in rows:
        if str(row.get("Stage", "")).strip().lower() == "after r3":
            return row
    return rows[-1]


def _float(row: dict[str, str], key: str) -> float:
    """安全解析指标列。"""

    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _mean_std(values: list[float]) -> tuple[float, float]:
    """返回总体均值和标准差。"""

    values = [value for value in values if not math.isnan(value)]
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return mean, std


def _fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 Stage 26 LoRa SSL + cross-day 三种子结果。")
    parser.add_argument("--root", default="results/stage26")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--seeds", default="7,13,31")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    records = []
    for seed_text in args.seeds.split(","):
        seed = int(seed_text.strip())
        save_dir = Path(args.root) / f"lora_ssl_plus_cross_day_seed{seed}_{args.job_id}"
        symbol = _r3(save_dir / "incremental_results.csv")
        recording = _r3(save_dir / "recording_level_incremental_results.csv")
        records.append(
            {
                "seed": seed,
                "overall": _float(symbol, "Overall Acc"),
                "old": _float(symbol, "Old Acc"),
                "new": _float(symbol, "New Acc"),
                "forgetting": _float(symbol, "Forgetting Rate"),
                "macro_f1": _float(symbol, "Macro F1"),
                "recording_overall": _float(recording, "Overall Acc"),
                "recording_new": _float(recording, "New Acc"),
            }
        )

    mean_std = {}
    for key in ("overall", "old", "new", "forgetting", "macro_f1", "recording_overall", "recording_new"):
        mean, std = _mean_std([float(record[key]) for record in records])
        mean_std[key] = {"mean": mean, "std": std}

    baseline = {
        "overall": 0.1603,
        "old": 0.0976,
        "new": 0.4111,
        "forgetting": 0.3325,
    }
    passed = (
        mean_std["overall"]["mean"] > baseline["overall"] + 0.01
        and mean_std["old"]["mean"] >= baseline["old"] - 0.01
        and mean_std["new"]["mean"] >= baseline["new"] - 0.03
        and mean_std["forgetting"]["mean"] <= baseline["forgetting"] + 0.05
    )
    verdict = (
        "通过三种子结构门槛，可作为 LoRa 当前正式候选。"
        if passed
        else "未通过三种子结构门槛，seed7 正信号不稳定。"
    )

    lines = [
        "# Stage 26 LoRa SSL + Cross-Day 三种子报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 方法：每轮 discovery 前执行 LoRa instance contrastive SSL，再接 cross-day 适配、GPCC、联合 discovery-CIL。",
        "- 对照：Stage 22 三种子 R3 Overall/Old/New/Forgetting = `0.1603/0.0976/0.4111/0.3325`。",
        "",
        "| Seed | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 | Rec Overall | Rec New |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['seed']} | {_fmt(record['overall'])} | {_fmt(record['old'])} | "
            f"{_fmt(record['new'])} | {_fmt(record['forgetting'])} | {_fmt(record['macro_f1'])} | "
            f"{_fmt(record['recording_overall'])} | {_fmt(record['recording_new'])} |"
        )
    lines.extend(["", "## 三种子均值 ± 标准差", ""])
    labels = {
        "overall": "R3 Overall",
        "old": "R3 Old",
        "new": "R3 New",
        "forgetting": "Forgetting",
        "macro_f1": "Macro F1",
        "recording_overall": "Recording Overall",
        "recording_new": "Recording New",
    }
    for key, label in labels.items():
        lines.append(f"- {label}: {_fmt(mean_std[key]['mean'])} ± {_fmt(mean_std[key]['std'])}")
    lines.extend(["", f"- 当前判定：{verdict}", ""])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "job_id": str(args.job_id),
        "records": records,
        "mean_std": mean_std,
        "baseline": baseline,
        "passed": passed,
        "verdict": verdict,
    }
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
