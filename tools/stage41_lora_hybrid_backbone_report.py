"""汇总 Stage 41 LoRa 门控双分支 backbone seed7 结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _read_r3(path: Path) -> dict[str, str]:
    """读取结果 CSV 中的 After R3 行。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if str(row.get("Stage", "")).strip().lower() == "after r3":
            return row
    return rows[-1]


def _metric(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _collect(root: Path, variant: str, job_id: str) -> dict[str, object]:
    save_dir = root / f"lora_{variant}_seed7_{job_id}"
    row = _read_r3(save_dir / "incremental_results.csv")
    result: dict[str, object] = {
        "variant": variant,
        "overall": _metric(row, "Overall Acc"),
        "old": _metric(row, "Old Acc"),
        "new": _metric(row, "New Acc"),
        "forgetting": _metric(row, "Forgetting Rate"),
    }
    recording_path = save_dir / "recording_level_incremental_results.csv"
    if recording_path.exists():
        recording = _read_r3(recording_path)
        result["recording_overall"] = _metric(recording, "Overall Acc")
        result["recording_new"] = _metric(recording, "New Acc")
    return result


def _fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 Stage 41 LoRa hybrid backbone 结果。")
    parser.add_argument("--root", default="results/stage41")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    baseline = _collect(root, "chirp", args.job_id)
    hybrid = _collect(root, "hybrid", args.job_id)
    delta = {
        key: float(hybrid[key]) - float(baseline[key])
        for key in ("overall", "old", "new", "forgetting")
    }
    passed = (
        delta["overall"] >= 0.02
        and delta["old"] >= -0.01
        and delta["new"] >= -0.03
    )
    verdict = "通过 seed7 结构门槛，可进入三种子确认。" if passed else "未通过 seed7 结构门槛，归档该候选。"
    lines = [
        "# Stage 41 LoRa 门控双分支 Backbone seed7 报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 对比：同一 strict split、GPCC、cross-day、LoRa SSL、联合 discovery-CIL 和 held-out 边界。",
        "- 新结构：原始 IQ Chirp 分支 + 幅度/相位几何分支 + 样本级门控融合。",
        "",
        "| Backbone | R3 Overall | R3 Old | R3 New | Forgetting | Recording Overall | Recording New |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in (baseline, hybrid):
        lines.append(
            f"| {item['variant']} | {_fmt(float(item['overall']))} | {_fmt(float(item['old']))} | "
            f"{_fmt(float(item['new']))} | {_fmt(float(item['forgetting']))} | "
            f"{_fmt(float(item.get('recording_overall', float('nan'))))} | "
            f"{_fmt(float(item.get('recording_new', float('nan'))))} |"
        )
    lines.extend([
        "",
        f"- Hybrid 相对 Chirp：Overall `{_fmt(delta['overall'])}`、Old `{_fmt(delta['old'])}`、"
        f"New `{_fmt(delta['new'])}`、Forgetting `{_fmt(delta['forgetting'])}`。",
        f"- 当前判定：{verdict}",
        "",
    ])
    summary = {
        "job_id": str(args.job_id),
        "baseline": baseline,
        "hybrid": hybrid,
        "delta_hybrid_minus_chirp": delta,
        "passed": passed,
        "verdict": verdict,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
