"""汇总 LoRa Chirp 表征下 RADCIL 与 DOI-style 后端的公平对照。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _rows(path: Path) -> list[dict[str, str]]:
    """读取结果 CSV，并在入口或实验失败时明确报错。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        result = list(csv.DictReader(handle))
    if not result:
        raise ValueError(f"结果为空：{path}")
    return result


def _r3(path: Path) -> dict[str, str]:
    """取 R3 结果行。"""

    result = _rows(path)
    for row in result:
        if str(row.get("Stage", "")).strip().lower() == "after r3":
            return row
    return result[-1]


def _value(row: dict[str, str], key: str) -> float:
    """解析指标；缺失值使用 NaN，避免误报为 0。"""

    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _fmt(value: float) -> str:
    """统一格式化报告数值。"""

    return "nan" if math.isnan(value) else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 LoRa Chirp 后端对照。")
    parser.add_argument("--root", default="results/stage28")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    save_dir = Path(args.root) / f"lora_chirp_backend_seed7_{args.job_id}"
    methods = {}
    for filename in ("incremental_results.csv", "shared_discovery_baseline_results.csv"):
        path = save_dir / filename
        if not path.exists():
            continue
        for row in [_r3(path)]:
            methods[str(row.get("Method", filename))] = {
                "source": filename,
                "overall": _value(row, "Overall Acc"),
                "old": _value(row, "Old Acc"),
                "new": _value(row, "New Acc"),
                "forgetting": _value(row, "Forgetting Rate"),
                "macro_f1": _value(row, "Macro F1"),
            }
    if not methods:
        raise ValueError(f"结果目录没有可用后端结果：{save_dir}")

    # DOI-style 只在同一 GPCC 伪标签下作为结构性参考，不把它包装成官方复现。
    reference = methods.get("DOI-style")
    radcil = methods.get("MV-ACC-CIL")
    delta = None
    if reference is not None and radcil is not None:
        delta = {
            key: float(reference[key]) - float(radcil[key])
            for key in ("overall", "old", "new", "forgetting")
        }

    lines = [
        "# Stage 28 LoRa Chirp 后端 seed7 对照报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 固定：LoRa Chirp backbone、strict split、GPCC、cross-day、LoRa SSL、联合 discovery-CIL。",
        "- 对照：同一批 discovery 伪标签下比较网络式 RADCIL 与 DOI-style 等后端。",
        "- DOI-style 说明：这是 DOI-inspired 简化 baseline，不是官方完整复现。",
        "",
        "| Backend | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 | Source |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, record in methods.items():
        lines.append(
            f"| {name} | {_fmt(float(record['overall']))} | {_fmt(float(record['old']))} | "
            f"{_fmt(float(record['new']))} | {_fmt(float(record['forgetting']))} | "
            f"{_fmt(float(record['macro_f1']))} | {record['source']} |"
        )
    if delta is not None:
        lines.extend([
            "",
            "## DOI-style 相对 RADCIL",
            "",
            f"- Overall `{_fmt(delta['overall'])}`、Old `{_fmt(delta['old'])}`、",
            f"New `{_fmt(delta['new'])}`、Forgetting `{_fmt(delta['forgetting'])}`。",
            "- 解释：若 DOI-style 明显提高 Old/Overall，说明 Chirp 表征已具备收益，剩余瓶颈主要在 RADCIL 的旧类分类边界；若两者都低，则继续看 discovery/表征质量。",
        ])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "job_id": str(args.job_id),
        "save_dir": str(save_dir),
        "methods": methods,
        "doi_minus_radcil": delta,
    }
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
