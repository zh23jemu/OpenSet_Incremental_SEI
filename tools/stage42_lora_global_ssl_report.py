"""汇总 Stage 42 LoRa 全局 discovery SSL seed7 结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASELINE = {"overall": 0.2581, "old": 0.1988, "new": 0.4952, "forgetting": 0.2738}


def _read_r3(save_dir: Path) -> dict[str, float]:
    frame = pd.read_csv(save_dir / "incremental_results.csv")
    rows = frame[frame["Stage"].astype(str).str.upper() == "AFTER R3"]
    row = rows.iloc[-1] if not rows.empty else frame.iloc[-1]
    return {
        "overall": float(row["Overall Acc"]),
        "old": float(row["Old Acc"]),
        "new": float(row["New Acc"]),
        "forgetting": float(row["Forgetting Rate"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 Stage 42 LoRa 全局 SSL 结果")
    parser.add_argument("--root", default="results/stage42")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    metrics = _read_r3(root / f"lora_global_ssl_seed7_{args.job_id}")
    delta = {key: metrics[key] - BASELINE[key] for key in BASELINE}
    passed = delta["overall"] >= 0.02 and delta["old"] >= -0.01 and delta["new"] >= -0.03
    lines = [
        "# Stage 42 LoRa 全局 discovery SSL Seed7 报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 预训练：Day1 IQ_1-6 + Day2-4 IQ_1-7 discovery 实例对比；不读取 IQ_8-10 held-out eval。",
        "",
        "| Variant | Overall | Old | New | Forgetting |",
        "|---|---:|---:|---:|---:|",
        f"| Chirp baseline | {BASELINE['overall']:.4f} | {BASELINE['old']:.4f} | {BASELINE['new']:.4f} | {BASELINE['forgetting']:.4f} |",
        f"| Global SSL | {metrics['overall']:.4f} | {metrics['old']:.4f} | {metrics['new']:.4f} | {metrics['forgetting']:.4f} |",
        "",
        f"- 相对 baseline：Overall `{delta['overall']:+.4f}`、Old `{delta['old']:+.4f}`、New `{delta['new']:+.4f}`。",
        f"- 判定：{'通过 seed7 门槛，可进入三种子确认。' if passed else '未通过 seed7 门槛，归档该方向。'}",
        "",
    ]
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps({"job_id": args.job_id, "baseline": BASELINE, "metrics": metrics, "delta": delta, "passed": passed}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
