"""生成 Stage 25 LoRa recording-level 评估聚合诊断报告。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


METRIC_KEYS = ("Overall Acc", "Old Acc", "New Acc", "Forgetting Rate", "Macro F1")


def _read_rows(path: Path) -> list[dict[str, str]]:
    """读取 CSV 结果，并在缺失或为空时给出明确错误。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"结果为空：{path}")
    return rows


def _float(row: dict[str, str], key: str) -> float:
    """把指标列转成 float；缺失值统一返回 NaN，避免报告器崩在空列上。"""

    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _fmt(value: float) -> str:
    """固定四位小数输出，便于和历史阶段报告横向对齐。"""

    return "nan" if math.isnan(value) else f"{value:.4f}"


def _stage_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """按 Stage 建索引，只保留主流程评估行。"""

    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        stage = str(row.get("Stage", "")).strip()
        if stage:
            indexed[stage] = row
    return indexed


def _metric_record(stage: str, symbol_row: dict[str, str], recording_row: dict[str, str]) -> dict[str, object]:
    """构造单个阶段的 symbol/recording 指标差值记录。"""

    record: dict[str, object] = {"stage": stage}
    for key in METRIC_KEYS:
        symbol_value = _float(symbol_row, key)
        recording_value = _float(recording_row, key)
        short_key = key.lower().replace(" ", "_")
        record[f"symbol_{short_key}"] = symbol_value
        record[f"recording_{short_key}"] = recording_value
        record[f"delta_{short_key}"] = recording_value - symbol_value
    record["source_eval_samples"] = int(float(recording_row.get("Source Eval Samples", "0") or 0))
    record["recording_groups"] = int(float(recording_row.get("Recording Groups", "0") or 0))
    return record


def build_report(save_dir: Path, job_id: str) -> tuple[str, dict[str, object]]:
    """读取一次实验输出，生成 Markdown 和结构化 summary。"""

    symbol_rows = _stage_rows(_read_rows(save_dir / "incremental_results.csv"))
    recording_rows = _stage_rows(_read_rows(save_dir / "recording_level_incremental_results.csv"))
    common_stages = [stage for stage in ("Initial", "After R1", "After R2", "After R3") if stage in symbol_rows and stage in recording_rows]
    if not common_stages:
        raise ValueError("symbol-level 与 recording-level 结果没有可对齐的 Stage。")

    records = [_metric_record(stage, symbol_rows[stage], recording_rows[stage]) for stage in common_stages]
    r3 = next((record for record in records if record["stage"] == "After R3"), records[-1])
    delta_overall = float(r3["delta_overall_acc"])
    delta_new = float(r3["delta_new_acc"])
    delta_old = float(r3["delta_old_acc"])
    verdict = (
        "recording-level 明显高于 symbol-level，LoRa 低分主要有评估粒度/逐 symbol 噪声放大的成分。"
        if delta_overall >= 0.05 and (math.isnan(delta_new) or delta_new >= -0.03)
        else "recording-level 没有明显抬高 Overall，LoRa 低分主要仍是表征/旧新类边界问题。"
    )

    lines = [
        "# Stage 25 LoRa Recording-Level 评估聚合诊断",
        "",
        f"- Slurm Job：`{job_id}`",
        f"- 输出目录：`{save_dir.as_posix()}`",
        "- 口径：训练、发现和伪标签注册完全不变；只在 held-out eval 结束后按 `recording_id` 做多数投票。",
        "- 边界：`recording_id` 是可观测元数据；真实标签只用于最后计算指标，不参与调参。",
        "",
        "| Stage | Symbol Overall | Recording Overall | ΔOverall | Symbol Old | Recording Old | ΔOld | Symbol New | Recording New | ΔNew | Recording Groups |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| {stage} | {sym_o} | {rec_o} | {delta_o} | {sym_old} | {rec_old} | {delta_old} | "
            "{sym_new} | {rec_new} | {delta_new} | {groups} |".format(
                stage=record["stage"],
                sym_o=_fmt(float(record["symbol_overall_acc"])),
                rec_o=_fmt(float(record["recording_overall_acc"])),
                delta_o=_fmt(float(record["delta_overall_acc"])),
                sym_old=_fmt(float(record["symbol_old_acc"])),
                rec_old=_fmt(float(record["recording_old_acc"])),
                delta_old=_fmt(float(record["delta_old_acc"])),
                sym_new=_fmt(float(record["symbol_new_acc"])),
                rec_new=_fmt(float(record["recording_new_acc"])),
                delta_new=_fmt(float(record["delta_new_acc"])),
                groups=record["recording_groups"],
            )
        )
    lines.extend(["", f"- R3 判定：{verdict}", ""])
    summary = {
        "job_id": str(job_id),
        "save_dir": str(save_dir),
        "records": records,
        "r3": r3,
        "verdict": verdict,
    }
    return "\n".join(lines), summary


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 Stage 25 LoRa recording-level 评估聚合诊断报告。")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    markdown, summary = build_report(Path(args.save_dir), args.job_id)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown + "\n", encoding="utf-8")
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Stage 25 LoRa recording-level 报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
