"""汇总 Stage 32 LoRa 类均衡分类头 + 新类 logits 保持结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VARIANTS = [
    ("base", 0, 0.0),
    ("e1_d0p5", 1, 0.5),
    ("e1_d1p0", 1, 1.0),
]


def _read_after_r3(path: Path) -> dict:
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
        "macro_f1": float(row["Macro F1"]),
    }


def _read_recording(path: Path) -> dict:
    if not path.exists():
        return {"recording_overall": float("nan"), "recording_new": float("nan")}
    frame = pd.read_csv(path)
    rows = frame[frame["Stage"].astype(str) == "After R3"]
    if rows.empty:
        return {"recording_overall": float("nan"), "recording_new": float("nan")}
    row = rows.iloc[-1].to_dict()
    return {
        "recording_overall": float(row["Overall Acc"]),
        "recording_new": float(row["New Acc"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/stage32")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    records = []
    for tag, epochs, distill in VARIANTS:
        save_dir = root / f"lora_balanced_head_distill_{tag}_{args.job_id}"
        record = {
            "tag": tag,
            "epochs": int(epochs),
            "new_distill_weight": float(distill),
        }
        record.update(_read_after_r3(save_dir / "incremental_results.csv"))
        record.update(_read_recording(save_dir / "recording_level_incremental_results.csv"))
        records.append(record)

    baseline = records[0]
    for record in records:
        record["delta_overall"] = record["overall"] - baseline["overall"]
        record["delta_old"] = record["old"] - baseline["old"]
        record["delta_new"] = record["new"] - baseline["new"]
        record["delta_forgetting"] = record["forgetting"] - baseline["forgetting"]
        record["passed"] = bool(
            record["tag"] != "base"
            and record["delta_overall"] >= 0.01
            and record["delta_old"] >= 0.02
            and record["delta_new"] >= -0.08
        )

    best = max(records, key=lambda item: item["overall"])
    payload = {
        "job_id": str(args.job_id),
        "records": records,
        "best_tag": best["tag"],
        "passed": bool(any(record["passed"] for record in records)),
    }
    Path(args.summary_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Stage 32 LoRa 类均衡分类头 + 新类保持 seed7 报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 固定结构：LoRa Chirp + GPCC + cross-day + LoRa SSL + joint discovery-CIL。",
        "- 变体：冻结 backbone 做类均衡 head 重校准，并对当前轮新伪类保留重校准前 logits 分布。",
        "- 判定门槛：Overall `+0.01`、Old `+0.02`，New 下降不超过 `0.08`。",
        "",
        "| Variant | Epochs | New distill | R3 Overall | R3 Old | R3 New | Forgetting | Rec Overall | Rec New | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for record in records:
        lines.append(
            "| {tag} | {epochs} | {new_distill_weight:.2f} | {overall:.4f} | {old:.4f} | "
            "{new:.4f} | {forgetting:.4f} | {recording_overall:.4f} | {recording_new:.4f} | {passed} |".format(
                **record
            )
        )
    lines.extend(
        [
            "",
            f"- 最佳 Overall：`{best['tag']}`，R3 Overall `{best['overall']:.4f}`。",
            f"- 当前判定：`{'通过，进入三种子确认' if payload['passed'] else '未通过，停止该候选'}`。",
        ]
    )
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
