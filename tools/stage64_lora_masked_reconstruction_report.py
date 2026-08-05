#!/usr/bin/env python3
"""汇总 Stage64 LoRa masked reconstruction 预训练 seed7 结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


STAGE48_SEED7 = {
    "overall": 0.2809523810,
    "old": 0.2220238095,
    "new": 0.5166666667,
    "forgetting": 0.1726190476,
    "recording_overall": 0.3066666667,
    "recording_new": 0.6666666667,
}


def _r3_row(path: Path) -> pd.Series:
    """读取最终 R3 行。"""
    if not path.exists():
        raise FileNotFoundError(f"Missing result CSV: {path}")
    frame = pd.read_csv(path)
    rows = frame.loc[frame["Stage"].astype(str) == "After R3"] if "Stage" in frame.columns else frame.tail(1)
    return rows.iloc[-1] if not rows.empty else frame.tail(1).iloc[0]


def _metric(row: pd.Series, name: str) -> float:
    """读取一个浮点指标。"""
    return float(row[name])


def collect(root: Path, job_id: str) -> dict[str, float | str]:
    """收集 Stage64 seed7 的 R3 指标。"""
    save_dir = root / f"lora_s28_masked_recon_seed7_{job_id}"
    symbol = _r3_row(save_dir / "incremental_results.csv")
    record: dict[str, float | str] = {
        "variant": "s28_masked_reconstruction_pretrain",
        "save_dir": str(save_dir),
        "overall": _metric(symbol, "Overall Acc"),
        "old": _metric(symbol, "Old Acc"),
        "new": _metric(symbol, "New Acc"),
        "forgetting": _metric(symbol, "Forgetting Rate"),
        "macro_f1": _metric(symbol, "Macro F1"),
    }
    recording_csv = save_dir / "recording_level_incremental_results.csv"
    if recording_csv.exists():
        recording = _r3_row(recording_csv)
        record["recording_overall"] = _metric(recording, "Overall Acc")
        record["recording_new"] = _metric(recording, "New Acc")
    for key, value in STAGE48_SEED7.items():
        if key in record:
            record[f"delta_stage48_{key}"] = float(record[key]) - float(value)
    return record


def passes_gate(record: dict[str, float | str]) -> bool:
    """预注册门槛：Overall/Old/New 至少形成平衡正收益。"""
    return (
        float(record["overall"]) >= STAGE48_SEED7["overall"] + 0.02
        and float(record["old"]) >= STAGE48_SEED7["old"]
        and float(record["new"]) >= STAGE48_SEED7["new"] - 0.02
    )


def fmt_optional(record: dict[str, float | str], key: str) -> str:
    """格式化可选指标。"""
    return f"{float(record[key]):.4f}" if key in record else "-"


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage64 LoRa masked reconstruction seed7 报告器")
    parser.add_argument("--root", default="results/stage64")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    job_id = str(args.job_id)
    record = collect(root, job_id)
    checkpoint_summary_path = root / f"stage64_lora_masked_reconstruction_checkpoint_summary_{job_id}.json"
    checkpoint_summary = json.loads(checkpoint_summary_path.read_text(encoding="utf-8"))
    gate = passes_gate(record)

    lines = [
        f"# Stage64 LoRa masked reconstruction 预训练 Seed7 报告（Job {job_id}）",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'通过 seed7 门槛，可进入三种子。' if gate else '未通过 seed7 门槛，不扩三种子。'}",
        f"- Stage48 seed7 对照 R3 Overall/Old/New/Forgetting = "
        f"{STAGE48_SEED7['overall']:.4f}/{STAGE48_SEED7['old']:.4f}/{STAGE48_SEED7['new']:.4f}/{STAGE48_SEED7['forgetting']:.4f}",
        f"- Stage64 R3 Overall/Old/New/Forgetting = "
        f"{float(record['overall']):.4f}/{float(record['old']):.4f}/{float(record['new']):.4f}/{float(record['forgetting']):.4f}",
        f"- Checkpoint 选择：fine-tune epoch `{checkpoint_summary['best']['epoch']}`，IQ_7 val acc `{checkpoint_summary['best']['validation_accuracy']:.4f}`。",
        f"- 预训练样本数：`{checkpoint_summary['pretrain_samples']}`，其中包含 discovery 未标注样本：`{checkpoint_summary['include_discovery_unlabeled']}`。",
        "",
        "## R3 指标",
        "",
        "| Variant | Overall | Old | New | Forgetting | ΔOverall | ΔOld | ΔNew | ΔForgetting | Recording Overall | Recording New | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        "| {variant} | {overall:.4f} | {old:.4f} | {new:.4f} | {forgetting:.4f} | "
        "{delta_overall:+.4f} | {delta_old:+.4f} | {delta_new:+.4f} | {delta_forgetting:+.4f} | "
        "{recording_overall} | {recording_new} | {gate} |".format(
            variant=record["variant"],
            overall=float(record["overall"]),
            old=float(record["old"]),
            new=float(record["new"]),
            forgetting=float(record["forgetting"]),
            delta_overall=float(record["delta_stage48_overall"]),
            delta_old=float(record["delta_stage48_old"]),
            delta_new=float(record["delta_stage48_new"]),
            delta_forgetting=float(record["delta_stage48_forgetting"]),
            recording_overall=fmt_optional(record, "recording_overall"),
            recording_new=fmt_optional(record, "recording_new"),
            gate="PASS" if gate else "FAIL",
        ),
        "",
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage64_lora_masked_reconstruction_summary_v1",
                "job_id": job_id,
                "stage48_seed7": STAGE48_SEED7,
                "record": record,
                "checkpoint": checkpoint_summary,
                "gate": "PASS" if gate else "FAIL",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
