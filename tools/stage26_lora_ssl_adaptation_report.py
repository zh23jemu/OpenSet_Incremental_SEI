"""汇总 Stage 26 LoRa 自监督 discovery 表征适配 seed7 结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    """读取 CSV，并在 Slurm 输出不完整时提前失败。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"结果为空：{path}")
    return rows


def _r3(path: Path) -> dict[str, str]:
    """读取 After R3 行；若缺失则使用最后一行并交给报告展示。"""

    rows = _read_rows(path)
    for row in rows:
        if str(row.get("Stage", "")).strip().lower() == "after r3":
            return row
    return rows[-1]


def _float(row: dict[str, str], key: str) -> float:
    """安全解析浮点指标。"""

    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def _collect_variant(root: Path, label: str, job_id: str) -> dict[str, object]:
    """收集单个 SSL 变体的 symbol-level 与 recording-level R3 指标。"""

    save_dir = root / f"lora_{label}_seed7_{job_id}"
    symbol = _r3(save_dir / "incremental_results.csv")
    record = {
        "label": label,
        "save_dir": str(save_dir),
        "symbol_overall": _float(symbol, "Overall Acc"),
        "symbol_old": _float(symbol, "Old Acc"),
        "symbol_new": _float(symbol, "New Acc"),
        "symbol_forgetting": _float(symbol, "Forgetting Rate"),
        "symbol_macro_f1": _float(symbol, "Macro F1"),
    }
    recording_path = save_dir / "recording_level_incremental_results.csv"
    if recording_path.exists():
        recording = _r3(recording_path)
        record.update(
            {
                "recording_overall": _float(recording, "Overall Acc"),
                "recording_old": _float(recording, "Old Acc"),
                "recording_new": _float(recording, "New Acc"),
                "recording_forgetting": _float(recording, "Forgetting Rate"),
                "recording_groups": int(float(recording.get("Recording Groups", "0") or 0)),
            }
        )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 Stage 26 LoRa SSL 表征适配 seed7 结果。")
    parser.add_argument("--root", default="results/stage26")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--labels", default="ssl_only,ssl_plus_cross_day")
    parser.add_argument("--baseline-overall", type=float, default=0.1752)
    parser.add_argument("--baseline-old", type=float, default=0.1036)
    parser.add_argument("--baseline-new", type=float, default=0.4619)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    records = [_collect_variant(root, label, args.job_id) for label in labels]
    best = max(records, key=lambda item: float(item["symbol_overall"]))
    delta_overall = float(best["symbol_overall"]) - float(args.baseline_overall)
    delta_new = float(best["symbol_new"]) - float(args.baseline_new)
    delta_old = float(best["symbol_old"]) - float(args.baseline_old)
    passed = delta_overall >= 0.02 and delta_old >= -0.01 and delta_new >= -0.03
    verdict = (
        "通过 seed7 结构门槛，可扩 seed13/31。"
        if passed
        else "未通过 seed7 结构门槛，不扩三种子；继续转向更强 LoRa backbone/预训练。"
    )

    lines = [
        "# Stage 26 LoRa SSL Discovery 表征适配 seed7 报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 对照：Stage 25 seed7 symbol-level R3 Overall/Old/New = "
        f"`{args.baseline_overall:.4f}/{args.baseline_old:.4f}/{args.baseline_new:.4f}`。",
        "- 边界：SSL 只用 Day1 已知训练标签和当前轮 discovery 无标签增强视图，不读取 held-out eval 或未知真值。",
        "",
        "| Variant | R3 Overall | ΔOverall | R3 Old | ΔOld | R3 New | ΔNew | Forgetting | Rec Overall | Rec New |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| {label} | {overall} | {delta_overall} | {old} | {delta_old} | {new} | {delta_new} | "
            "{forgetting} | {rec_overall} | {rec_new} |".format(
                label=record["label"],
                overall=_fmt(float(record["symbol_overall"])),
                delta_overall=_fmt(float(record["symbol_overall"]) - float(args.baseline_overall)),
                old=_fmt(float(record["symbol_old"])),
                delta_old=_fmt(float(record["symbol_old"]) - float(args.baseline_old)),
                new=_fmt(float(record["symbol_new"])),
                delta_new=_fmt(float(record["symbol_new"]) - float(args.baseline_new)),
                forgetting=_fmt(float(record["symbol_forgetting"])),
                rec_overall=_fmt(float(record.get("recording_overall", float("nan")))),
                rec_new=_fmt(float(record.get("recording_new", float("nan")))),
            )
        )
    lines.extend(
        [
            "",
            f"- 最佳变体：`{best['label']}`，R3 Overall 改变量 `{_fmt(delta_overall)}`。",
            f"- 当前判定：{verdict}",
            "",
        ]
    )

    summary = {
        "job_id": str(args.job_id),
        "baseline": {
            "overall": float(args.baseline_overall),
            "old": float(args.baseline_old),
            "new": float(args.baseline_new),
        },
        "records": records,
        "best": best,
        "passed": passed,
        "verdict": verdict,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Stage 26 LoRa SSL 报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
