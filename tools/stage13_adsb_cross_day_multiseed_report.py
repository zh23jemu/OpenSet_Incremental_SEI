"""汇总 ADS-B 跨天表征适配三种子完整 CIL 结果。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _last_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"结果为空：{path}")
    return rows[-1]


def _number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _mean_std(values: list[float]) -> tuple[float, float]:
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def _fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 ADS-B 跨天表征适配三种子结果。")
    parser.add_argument("--root", default="results/stage13")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for seed in (7, 13, 31):
        path = Path(args.root) / f"adsb_cross_day_seed{seed}_{args.job_id}" / "incremental_results.csv"
        row = _last_row(path)
        rows.append(
            {
                "seed": seed,
                "overall": _number(row, "Overall Acc"),
                "old": _number(row, "Old Acc"),
                "new": _number(row, "New Acc"),
                "forgetting": _number(row, "Forgetting"),
                "macro_f1": _number(row, "Macro F1"),
            }
        )

    lines = [
        "# Stage 13 ADS-B 跨天表征适配三种子报告",
        "",
        "组合：cross_day 训练期表征适配 + GPCC discovery + 当前 RADCIL 后端。",
        "",
        "| Seed | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {_fmt(row['overall'])} | {_fmt(row['old'])} | "
            f"{_fmt(row['new'])} | {_fmt(row['forgetting'])} | {_fmt(row['macro_f1'])} |"
        )
    lines.append("")
    for key, label in (
        ("overall", "R3 Overall"),
        ("old", "R3 Old"),
        ("new", "R3 New"),
        ("forgetting", "Forgetting"),
        ("macro_f1", "Macro F1"),
    ):
        mean, std = _mean_std([row[key] for row in rows])
        lines.append(f"- {label}: {_fmt(mean)} ± {_fmt(std)}")
    lines.extend(
        [
            "",
            "判定：与同 seed 的 ADS-B target split baseline 对照；只有三种子 Overall 稳定提升且 Old/New 不出现明显塌缩，才锁定为正式候选。",
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
