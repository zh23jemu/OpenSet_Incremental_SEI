"""汇总 Stage 43 LoRa recording-level hybrid 三种子结果。

Stage 41 说明 hybrid 在 symbol-level 会压低 New，但 seed7 的
recording-level Overall 反而高于 Chirp。这里不再修改训练机制，只验证
如果客户/论文接受 recording-level 任务口径，hybrid 的 recording 聚合收益
是否能跨 seed 稳定存在。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd


CHIRP_RECORDING_BASELINE = {
    "overall": 0.2933,
    "new": 0.6444,
}


def _r3(path: Path) -> dict[str, float]:
    """读取某个增量结果 CSV 的 After R3 行。"""
    frame = pd.read_csv(path)
    rows = frame[frame["Stage"].astype(str).str.upper() == "AFTER R3"]
    row = rows.iloc[-1] if not rows.empty else frame.iloc[-1]
    return {
        "overall": float(row["Overall Acc"]),
        "old": float(row.get("Old Acc", 0.0)),
        "new": float(row.get("New Acc", 0.0)),
        "forgetting": float(row.get("Forgetting Rate", 0.0)),
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    """返回三种子均值和样本标准差，单种子时标准差置 0。"""
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 Stage 43 LoRa recording-level hybrid 三种子结果")
    parser.add_argument("--root", default="results/stage43")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--seeds", default="7,13,31")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    records: list[dict[str, object]] = []
    for seed in seeds:
        save_dir = root / f"lora_hybrid_recording_seed{seed}_{args.job_id}"
        symbol = _r3(save_dir / "incremental_results.csv")
        recording = _r3(save_dir / "recording_level_incremental_results.csv")
        records.append({"seed": seed, "symbol": symbol, "recording": recording, "save_dir": str(save_dir)})

    summary = {
        metric: _mean_std([float(item["recording"][metric]) for item in records])
        for metric in ("overall", "old", "new", "forgetting")
    }
    symbol_summary = {
        metric: _mean_std([float(item["symbol"][metric]) for item in records])
        for metric in ("overall", "old", "new", "forgetting")
    }
    delta_vs_chirp_recording = {
        "overall": summary["overall"]["mean"] - CHIRP_RECORDING_BASELINE["overall"],
        "new": summary["new"]["mean"] - CHIRP_RECORDING_BASELINE["new"],
    }
    passed = (
        delta_vs_chirp_recording["overall"] >= 0.02
        and delta_vs_chirp_recording["new"] >= -0.08
    )

    lines = [
        "# Stage 43 LoRa Recording-Level Hybrid 三种子报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 目标：验证 recording-level 任务口径下，hybrid 的 seed7 recording 收益是否稳定。",
        "- 边界：strict split 不变；IQ_8-10 只用于 held-out evaluation；不使用未知真值调参。",
        "",
        "| Seed | Symbol Overall | Symbol Old | Symbol New | Recording Overall | Recording Old | Recording New |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in records:
        symbol = item["symbol"]
        recording = item["recording"]
        lines.append(
            f"| {item['seed']} | {symbol['overall']:.4f} | {symbol['old']:.4f} | {symbol['new']:.4f} | "
            f"{recording['overall']:.4f} | {recording['old']:.4f} | {recording['new']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- Hybrid recording Overall：`{summary['overall']['mean']:.4f}±{summary['overall']['std']:.4f}`。",
            f"- Hybrid recording New：`{summary['new']['mean']:.4f}±{summary['new']['std']:.4f}`。",
            f"- 对照 Stage 27 Chirp recording：Overall `{CHIRP_RECORDING_BASELINE['overall']:.4f}`，New `{CHIRP_RECORDING_BASELINE['new']:.4f}`。",
            f"- 相对 Chirp recording：Overall `{delta_vs_chirp_recording['overall']:+.4f}`，New `{delta_vs_chirp_recording['new']:+.4f}`。",
            f"- 判定：{'通过 recording-level 三种子门槛。' if passed else '未通过 recording-level 三种子门槛，归档该方向。'}",
            "",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "job_id": str(args.job_id),
                "records": records,
                "recording_summary": summary,
                "symbol_summary": symbol_summary,
                "chirp_recording_baseline": CHIRP_RECORDING_BASELINE,
                "delta_vs_chirp_recording": delta_vs_chirp_recording,
                "passed": passed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
