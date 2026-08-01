"""汇总 Stage 31 LoRa 类均衡分类头重校准 seed7 结果。

报告器只读取每个变体的 CSV/JSON 小型结果，不读取 held-out 真值做选择；
训练脚本已经完成最终评估，这里只负责按配置汇总并给出预注册门槛判定。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _latest_row(path: Path) -> dict:
    frame = pd.read_csv(path)
    rows = frame[frame["Stage"].astype(str) == "After R3"]
    if rows.empty:
        raise ValueError(f"{path} 缺少 After R3 行")
    row = rows.iloc[-1].to_dict()
    return {
        "overall": float(row["Overall Acc"]),
        "old": float(row["Old Acc"]),
        "new": float(row["New Acc"]),
        "forgetting": float(row["Forgetting Rate"]),
    }


def _recording_row(path: Path) -> dict:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    rows = frame[frame["Stage"].astype(str) == "After R3"]
    if rows.empty:
        return {}
    row = rows.iloc[-1].to_dict()
    return {
        "recording_overall": float(row.get("Overall Acc", float("nan"))),
        "recording_new": float(row.get("New Acc", float("nan"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/stage31")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    records = []
    for epochs in (0, 1, 3):
        save_dir = root / f"lora_balanced_head_seed7_e{epochs}_{args.job_id}"
        metrics = _latest_row(save_dir / "incremental_results.csv")
        metrics.update(_recording_row(save_dir / "recording_level_incremental_results.csv"))
        records.append({"epochs": epochs, **metrics})

    baseline = records[0]
    for record in records:
        record["delta_overall"] = record["overall"] - baseline["overall"]
        record["delta_old"] = record["old"] - baseline["old"]
        record["delta_new"] = record["new"] - baseline["new"]
        record["delta_forgetting"] = record["forgetting"] - baseline["forgetting"]
        record["passed"] = bool(
            record["delta_overall"] >= 0.01
            and record["delta_old"] >= 0.02
            and record["delta_new"] >= -0.08
        )

    best = max(records, key=lambda item: item["overall"])
    payload = {
        "job_id": str(args.job_id),
        "records": records,
        "best_epochs": int(best["epochs"]),
        "passed": bool(any(item["passed"] for item in records if item["epochs"] > 0)),
    }
    Path(args.summary_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# Stage 31 LoRa 类均衡分类头重校准 seed7 报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 固定结构：LoRa Chirp + GPCC + cross-day + LoRa SSL + joint discovery-CIL。",
        "- 变体：每轮 CIL 后冻结 backbone，按类均衡重校准 classifier；0 为历史基线。",
        "- 选择边界：只使用训练/replay 与 discovery 伪标签；held-out IQ_8-10 只用于最终评估。",
        "",
        "| Recalibration epochs | R3 Overall | R3 Old | R3 New | Forgetting | Rec Overall | Rec New | Pass |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for record in records:
        lines.append(
            "| {epochs} | {overall:.4f} | {old:.4f} | {new:.4f} | {forgetting:.4f} | "
            "{recording_overall:.4f} | {recording_new:.4f} | {passed} |".format(
                recording_overall=record.get("recording_overall", float("nan")),
                recording_new=record.get("recording_new", float("nan")),
                **record,
            )
        )
    lines.extend([
        "",
        f"- 最佳 Overall：`epochs={best['epochs']}`，R3 Overall `{best['overall']:.4f}`。",
        f"- 当前判定：`{'通过，进入三种子确认' if payload['passed'] else '未通过，停止该候选'}`。",
    ])
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
