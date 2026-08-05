#!/usr/bin/env python3
"""汇总 Stage55 LoRa raw s28 物理域预训练 seed7 结果。

Stage55 只替换初始 LoRa Chirp checkpoint：在当前正式 raw s28 数据口径下，
用 Day1 IQ_1-6 / IQ_7 训练物理描述符辅助目标，再接 Stage48 rec065 主流程。
报告器只读取已经生成的 checkpoint summary 和 held-out CSV，不参与训练或选参。
"""

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
    """读取最终 R3 行，兼容 Stage/Round 两种输出格式。"""
    if not path.exists():
        raise FileNotFoundError(f"Missing result CSV: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty result CSV: {path}")
    if "Round" in frame.columns:
        row = frame.loc[frame["Round"] == 3]
    elif "Stage" in frame.columns:
        normalized = frame["Stage"].astype(str).str.strip().str.lower()
        row = frame.loc[normalized.isin({"after r3", "r3"})]
    else:
        raise KeyError(f"{path} has neither Round nor Stage column: {list(frame.columns)}")
    if row.empty:
        raise ValueError(f"Missing R3 row in {path}")
    return row.iloc[-1]


def _metric(row: pd.Series, name: str) -> float:
    """把 pandas 标量统一成普通 float，便于 JSON 序列化。"""
    return float(row[name])


def collect(root: Path, job_id: str) -> dict[str, float | str]:
    """收集 Stage55 seed7 的 symbol-level 与 recording-level R3 指标。"""
    save_dir = root / f"lora_s28_physical_seed7_{job_id}"
    symbol = _r3_row(save_dir / "incremental_results.csv")
    record: dict[str, float | str] = {
        "variant": "s28_physical_pretrain",
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
    """预注册门槛：必须提高 Overall，并保住 Old/New/Forgetting。"""
    return (
        float(record["overall"]) >= STAGE48_SEED7["overall"] + 0.01
        and float(record["old"]) >= STAGE48_SEED7["old"]
        and float(record["new"]) >= STAGE48_SEED7["new"] - 0.02
        and float(record["forgetting"]) <= STAGE48_SEED7["forgetting"] + 0.02
    )


def fmt_optional(record: dict[str, float | str], key: str) -> str:
    """格式化可选 recording-level 指标。"""
    return f"{float(record[key]):.4f}" if key in record else "-"


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage55 LoRa raw s28 physical pretrain seed7 报告器")
    parser.add_argument("--root", default="results/stage55")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    job_id = str(args.job_id)
    record = collect(root, job_id)
    checkpoint_summary_path = root / f"stage55_lora_s28_physical_checkpoint_summary_{job_id}.json"
    checkpoint_summary = json.loads(checkpoint_summary_path.read_text(encoding="utf-8"))
    gate = passes_gate(record)

    lines = [
        f"# Stage55 LoRa raw s28 物理域预训练 Seed7 报告（Job {job_id}）",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'通过 seed7 门槛，可进入三种子。' if gate else '未通过 seed7 门槛，不扩三种子。'}",
        f"- Stage48 seed7 raw s28+rec065 对照 R3 Overall/Old/New/Forgetting = "
        f"{STAGE48_SEED7['overall']:.4f}/{STAGE48_SEED7['old']:.4f}/{STAGE48_SEED7['new']:.4f}/{STAGE48_SEED7['forgetting']:.4f}",
        f"- Stage55 R3 Overall/Old/New/Forgetting = "
        f"{float(record['overall']):.4f}/{float(record['old']):.4f}/{float(record['new']):.4f}/{float(record['forgetting']):.4f}",
        f"- Checkpoint 选择：epoch `{checkpoint_summary['best']['epoch']}`，IQ_7 val acc `{checkpoint_summary['best']['validation_accuracy']:.4f}`。",
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
        "## 说明",
        "",
        "- checkpoint 训练只使用 Day1 IQ_1-6 train 和 IQ_7 validation，不读取 IQ_8-10 held-out eval。",
        "- 后续 CIL 固定 Stage48 raw s28 + recording-consensus 0.65 主配置，不改变后端倍率或聚类阈值。",
        "- 若 seed7 不过门槛，不继续三种子，避免把物理描述符目标变成新一轮小权重搜索。",
        "",
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage55_lora_s28_physical_pretrain_summary_v1",
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
