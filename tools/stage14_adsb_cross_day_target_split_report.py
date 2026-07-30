"""汇总 Stage 14 ADS-B cross_day + target split seed7 结果。"""

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


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 ADS-B cross_day + target split 结果。")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    row = _last_row(Path(args.save_dir) / "incremental_results.csv")
    fields = ("Overall Acc", "Old Acc", "New Acc", "Forgetting", "Macro F1")
    lines = [
        "# Stage 14 ADS-B cross_day + target split seed7 报告",
        "",
        "组合：cross_day 训练期表征适配 + MV-ACC target split + 当前 RADCIL 后端。",
        "",
        "| R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 |",
        "|---:|---:|---:|---:|---:|",
        "| " + " | ".join(row.get(field, "nan") for field in fields) + " |",
        "",
        "判定：与 Stage 12/13 的 cross_day + GPCC 以及 target split baseline 对照；"
        "只有 Overall 超过 0.50 且 Old/New 不塌缩，才进入三种子确认。",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
