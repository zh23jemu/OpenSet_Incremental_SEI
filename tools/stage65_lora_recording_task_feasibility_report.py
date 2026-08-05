#!/usr/bin/env python3
"""Stage65：LoRa recording/transmission-level 任务口径可行性汇总。

本工具不训练模型，只读取已经完成的 strict held-out eval 结果。它比较
symbol-level 与 recording-level majority vote 指标，用来回答一个实际问题：
如果把 LoRa 任务从逐 symbol 识别改成更贴近采集过程的 recording/transmission
级识别，当前最佳候选是否能接近客户期望的 50%。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _r3(path: Path) -> pd.Series:
    """读取一个 CSV 的 R3/After R3 行。"""
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    frame = pd.read_csv(path)
    if "Stage" in frame.columns:
        rows = frame.loc[frame["Stage"].astype(str) == "After R3"]
        return rows.iloc[-1] if not rows.empty else frame.tail(1).iloc[0]
    if "Round" in frame.columns:
        rows = frame.loc[frame["Round"].astype(str) == "R3"]
        return rows.iloc[-1] if not rows.empty else frame.tail(1).iloc[0]
    return frame.tail(1).iloc[0]


def _collect_one(label: str, save_dir: Path, seed: int | None = None) -> dict[str, float | int | str]:
    """读取一个实验目录的 symbol-level 与 recording-level R3 指标。"""
    symbol = _r3(save_dir / "incremental_results.csv")
    recording = _r3(save_dir / "recording_level_incremental_results.csv")
    return {
        "label": label,
        "seed": -1 if seed is None else int(seed),
        "save_dir": str(save_dir),
        "symbol_overall": float(symbol["Overall Acc"]),
        "symbol_old": float(symbol["Old Acc"]),
        "symbol_new": float(symbol["New Acc"]),
        "symbol_forgetting": float(symbol["Forgetting Rate"]),
        "recording_overall": float(recording["Overall Acc"]),
        "recording_old": float(recording["Old Acc"]),
        "recording_new": float(recording["New Acc"]),
        "recording_forgetting": float(recording["Forgetting Rate"]),
        "recording_groups": int(recording["Recording Groups"]),
    }


def _mean_std(records: list[dict[str, float | int | str]], key: str) -> str:
    """格式化均值和标准差。"""
    values = np.asarray([float(item[key]) for item in records], dtype=np.float64)
    return f"{values.mean():.4f}±{values.std(ddof=0):.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage65 LoRa recording-level task feasibility report")
    parser.add_argument("--root", default="results")
    parser.add_argument("--output", default="results/stage65/STAGE65_LORA_RECORDING_TASK_FEASIBILITY_REPORT.md")
    parser.add_argument("--summary-json", default="results/stage65/stage65_lora_recording_task_feasibility_summary.json")
    args = parser.parse_args()

    root = Path(args.root)
    records: list[dict[str, float | int | str]] = []
    for seed in (7, 13, 31):
        records.append(
            _collect_one(
                "Stage48 raw s28 rec065",
                root / "stage48" / f"lora_s28_rec065_seed{seed}_46124799",
                seed=seed,
            )
        )
    records.append(
        _collect_one(
            "Stage63 consistency pretrain",
            root / "stage63" / "lora_s28_consistency_seed7_46319406",
            seed=7,
        )
    )
    records.append(
        _collect_one(
            "Stage64 masked reconstruction",
            root / "stage64" / "lora_s28_masked_recon_seed7_46321233",
            seed=7,
        )
    )

    stage48 = [item for item in records if item["label"] == "Stage48 raw s28 rec065"]
    best_recording = max(records, key=lambda item: float(item["recording_overall"]))
    best_symbol = max(records, key=lambda item: float(item["symbol_overall"]))
    recording_close_to_target = float(best_recording["recording_overall"]) >= 0.50

    lines = [
        "# Stage65 LoRa Recording/Transmission-Level 任务口径可行性报告",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'recording-level 已接近 50%，可考虑任务口径切换。' if recording_close_to_target else 'recording-level 仍明显低于 50%，不能靠改评估粒度解决客户低分问题。'}",
        f"- Stage48 三种子 recording-level R3 Overall/Old/New/Forgetting = "
        f"{_mean_std(stage48, 'recording_overall')}/{_mean_std(stage48, 'recording_old')}/{_mean_std(stage48, 'recording_new')}/{_mean_std(stage48, 'recording_forgetting')}。",
        f"- 最好 recording-level Overall：{best_recording['label']} seed {best_recording['seed']}，Overall={float(best_recording['recording_overall']):.4f}，New={float(best_recording['recording_new']):.4f}。",
        f"- 最好 symbol-level Overall：{best_symbol['label']} seed {best_symbol['seed']}，Overall={float(best_symbol['symbol_overall']):.4f}，New={float(best_symbol['symbol_new']):.4f}。",
        "",
        "## R3 对比表",
        "",
        "| Label | Seed | Symbol Overall | Symbol Old | Symbol New | Symbol Forgetting | Recording Overall | Recording Old | Recording New | Recording Forgetting | Groups |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in records:
        lines.append(
            "| {label} | {seed} | {so:.4f} | {sold:.4f} | {sn:.4f} | {sf:.4f} | {ro:.4f} | {rold:.4f} | {rn:.4f} | {rf:.4f} | {groups} |".format(
                label=item["label"],
                seed=int(item["seed"]),
                so=float(item["symbol_overall"]),
                sold=float(item["symbol_old"]),
                sn=float(item["symbol_new"]),
                sf=float(item["symbol_forgetting"]),
                ro=float(item["recording_overall"]),
                rold=float(item["recording_old"]),
                rn=float(item["recording_new"]),
                rf=float(item["recording_forgetting"]),
                groups=int(item["recording_groups"]),
            )
        )
    lines.extend(
        [
            "",
            "## 边界说明",
            "",
            "- recording-level 只在 held-out eval 结束后按可观测 `recording_id` 做多数投票；不参与训练、聚类、伪标签注册或模型选择。",
            "- 因为最好 recording-level Overall 仍只有约 0.36，LoRa 当前不能通过改评估口径包装成 50% 以上结果。",
            "- 后续如果继续研究，应重新定义更窄的 transmission-level 任务，或面向客户/论文诚实报告 LoRa 跨天跨体制局限。",
            "",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage65_lora_recording_task_feasibility_v1",
                "records": records,
                "stage48_recording": {
                    "overall": _mean_std(stage48, "recording_overall"),
                    "old": _mean_std(stage48, "recording_old"),
                    "new": _mean_std(stage48, "recording_new"),
                    "forgetting": _mean_std(stage48, "recording_forgetting"),
                },
                "best_recording": best_recording,
                "best_symbol": best_symbol,
                "recording_close_to_target": bool(recording_close_to_target),
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
