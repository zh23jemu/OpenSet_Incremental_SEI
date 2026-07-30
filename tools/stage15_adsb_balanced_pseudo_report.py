"""汇总 Stage 15 ADS-B 类均衡伪标签训练 seed7 结果。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _last_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"结果为空：{path}")
    return rows[-1]


def _metric(row: dict[str, str], *keys: str) -> str:
    """兼容不同实验阶段的指标列名，优先返回第一个存在且非空的字段。"""
    for key in keys:
        if key in row and str(row[key]).strip() != "":
            return row[key]
    return "nan"


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 ADS-B 类均衡伪标签训练 seed7 结果。")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    row = _last_row(Path(args.save_dir) / "incremental_results.csv")
    lines = [
        "# Stage 15 ADS-B 类均衡伪标签训练 seed7 报告",
        "",
        "组合：cross_day 训练期表征适配 + GPCC discovery + 类均衡伪标签 RADCIL。",
        "",
        "| R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 |",
        "|---:|---:|---:|---:|---:|",
        (
            f"| {_metric(row, 'Overall Acc')} | {_metric(row, 'Old Acc')} | "
            f"{_metric(row, 'New Acc')} | {_metric(row, 'Forgetting', 'Forgetting Rate')} | "
            f"{_metric(row, 'Macro F1')} |"
        ),
        "",
        "判定：目标是相对 Stage 12/13 的 cross_day + GPCC seed7 继续提升 Overall，且 Old 不明显下降。",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
