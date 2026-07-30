"""汇总 Stage 12 跨天表征适配完整 CIL seed7 结果。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_last_row(path: Path) -> dict[str, str]:
    """读取 incremental_results.csv 的最后一轮，作为 R3 汇总。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"结果为空：{path}")
    return rows[-1]


def _value(row: dict[str, str], key: str) -> str:
    return row.get(key, "nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 Stage 12 完整 CIL 结果。")
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--adsb-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    lora = _read_last_row(Path(args.lora_dir) / "incremental_results.csv")
    adsb = _read_last_row(Path(args.adsb_dir) / "incremental_results.csv")
    lines = [
        "# Stage 12 跨天表征适配完整 CIL seed7 报告",
        "",
        "组合：GPCC discovery + cross_day 训练期表征适配 + 当前 RADCIL 后端。",
        "本报告只读取最终 R3 incremental_results.csv；held-out eval 只用于最终评估，不参与适配或发现决策。",
        "",
        "| 数据集 | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| LoRa25 | {_value(lora, 'Overall Acc')} | {_value(lora, 'Old Acc')} | {_value(lora, 'New Acc')} | {_value(lora, 'Forgetting')} | {_value(lora, 'Macro F1')} |",
        f"| ADS-B | {_value(adsb, 'Overall Acc')} | {_value(adsb, 'Old Acc')} | {_value(adsb, 'New Acc')} | {_value(adsb, 'Forgetting')} | {_value(adsb, 'Macro F1')} |",
        "",
        "判定：与各数据集已有 seed7 主后端结果比较；只有 Overall 明显提升且 Old/New 不出现塌缩，才考虑扩展 seed13/31。",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
