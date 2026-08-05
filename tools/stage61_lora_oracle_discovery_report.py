#!/usr/bin/env python3
"""汇总 Stage61 LoRa oracle discovery 上界诊断结果。

该报告明确标注 oracle discovery 读取当前 discovery 真值，仅用于瓶颈定位。
它不能作为正式方法、baseline 或论文主表结果。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


STAGE48_MULTISEED = {
    "overall": 0.2803,
    "old": 0.2312,
    "new": 0.4770,
    "forgetting": 0.1802,
}


def _r3(path: Path) -> pd.Series:
    """读取最终 R3 行。"""
    frame = pd.read_csv(path)
    row = frame.loc[frame["Stage"].astype(str) == "After R3"]
    if row.empty:
        row = frame.tail(1)
    return row.iloc[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage61 LoRa oracle discovery 上界报告器")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    r3 = _r3(save_dir / "incremental_results.csv")
    record = {
        "overall": float(r3["Overall Acc"]),
        "old": float(r3["Old Acc"]),
        "new": float(r3["New Acc"]),
        "forgetting": float(r3["Forgetting Rate"]),
        "macro_f1": float(r3["Macro F1"]),
    }
    gate = (
        record["overall"] >= STAGE48_MULTISEED["overall"] + 0.10
        and record["new"] >= STAGE48_MULTISEED["new"]
    )
    summary = {
        "schema_version": "stage61_lora_oracle_discovery_summary_v1",
        "job_id": str(args.job_id),
        "save_dir": str(save_dir),
        "oracle_warning": "uses current discovery true labels; diagnostic upper bound only",
        "r3": record,
        "stage48_multiseed": STAGE48_MULTISEED,
        "diagnostic_gate": "DISCOVERY_LIMITED" if gate else "NOT_DISCOVERY_ONLY",
    }
    (save_dir / "stage61_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        f"# Stage61 LoRa Oracle Discovery 上界诊断（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- R3 Overall/Old/New/Forgetting = {record['overall']:.4f}/{record['old']:.4f}/{record['new']:.4f}/{record['forgetting']:.4f}。",
        f"- 诊断判定：{summary['diagnostic_gate']}。",
        "- 注意：该实验读取当前 discovery 真值生成完美簇，只能用于定位瓶颈，不能作为正式方法。",
        "",
        "## R3 对比",
        "",
        "| Setting | Overall | Old | New | Forgetting |",
        "|---|---:|---:|---:|---:|",
        f"| Stage61 oracle discovery | {record['overall']:.4f} | {record['old']:.4f} | {record['new']:.4f} | {record['forgetting']:.4f} |",
        f"| Stage48 multiseed | {STAGE48_MULTISEED['overall']:.4f} | {STAGE48_MULTISEED['old']:.4f} | {STAGE48_MULTISEED['new']:.4f} | {STAGE48_MULTISEED['forgetting']:.4f} |",
        "",
    ]
    (save_dir / f"STAGE61_LORA_ORACLE_DISCOVERY_REPORT_{args.job_id}.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote Stage61 report in {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
