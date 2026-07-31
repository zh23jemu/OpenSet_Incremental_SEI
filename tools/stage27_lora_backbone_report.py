"""汇总 Stage 27 LoRa-specific backbone seed7 对比结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    """读取增量结果 CSV。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"结果为空：{path}")
    return rows


def _r3(path: Path) -> dict[str, str]:
    """读取 After R3 行。"""
    rows = _read_rows(path)
    for row in rows:
        if str(row.get("Stage", "")).strip().lower() == "after r3":
            return row
    return rows[-1]


def _float(row: dict[str, str], key: str) -> float:
    """安全解析指标。"""
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def _collect(root: Path, variant: str, job_id: str) -> dict[str, object]:
    """收集单个 backbone 的 symbol/recording R3 指标。"""
    save_dir = root / f"lora_{variant}_seed7_{job_id}"
    symbol = _r3(save_dir / "incremental_results.csv")
    record: dict[str, object] = {
        "variant": variant,
        "save_dir": str(save_dir),
        "overall": _float(symbol, "Overall Acc"),
        "old": _float(symbol, "Old Acc"),
        "new": _float(symbol, "New Acc"),
        "forgetting": _float(symbol, "Forgetting Rate"),
        "macro_f1": _float(symbol, "Macro F1"),
    }
    recording_path = save_dir / "recording_level_incremental_results.csv"
    if recording_path.exists():
        recording = _r3(recording_path)
        record["recording_overall"] = _float(recording, "Overall Acc")
        record["recording_new"] = _float(recording, "New Acc")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 Stage 27 LoRa backbone seed7 对比。")
    parser.add_argument("--root", default="results/stage27")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    records = [_collect(root, "resnet1d", args.job_id), _collect(root, "chirp", args.job_id)]
    baseline = next(record for record in records if record["variant"] == "resnet1d")
    chirp = next(record for record in records if record["variant"] == "chirp")
    delta = {
        key: float(chirp[key]) - float(baseline[key])
        for key in ("overall", "old", "new", "forgetting", "macro_f1")
    }
    passed = delta["overall"] >= 0.02 and delta["old"] >= -0.01 and delta["new"] >= -0.03
    verdict = "通过 seed7 结构门槛，可进入三种子确认。" if passed else "未通过 seed7 结构门槛，不扩三种子。"

    lines = [
        "# Stage 27 LoRa Chirp Backbone seed7 报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 对比：同一 strict split、同一 GPCC + cross-day + 联合 discovery-CIL 后端，只替换 Day1 closed-set backbone。",
        "- 边界：不读取 held-out eval 真值进行训练或选择。",
        "",
        "| Backbone | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 | Recording Overall | Recording New |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['variant']} | {_fmt(float(record['overall']))} | {_fmt(float(record['old']))} | "
            f"{_fmt(float(record['new']))} | {_fmt(float(record['forgetting']))} | {_fmt(float(record['macro_f1']))} | "
            f"{_fmt(float(record.get('recording_overall', float('nan'))))} | "
            f"{_fmt(float(record.get('recording_new', float('nan'))))} |"
        )
    lines.extend([
        "",
        "- Chirp 相对 ResNet1D："
        f" Overall `{_fmt(delta['overall'])}`、Old `{_fmt(delta['old'])}`、"
        f"New `{_fmt(delta['new'])}`、Forgetting `{_fmt(delta['forgetting'])}`。",
        f"- 当前判定：{verdict}",
        "",
    ])
    summary = {
        "job_id": str(args.job_id),
        "records": records,
        "delta_chirp_minus_resnet1d": delta,
        "passed": passed,
        "verdict": verdict,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
