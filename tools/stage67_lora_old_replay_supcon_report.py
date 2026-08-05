#!/usr/bin/env python3
"""Stage67：LoRa 旧类 replay SupCon seed7 小矩阵报告器。

Stage66 已经把 LoRa 冲 50% 的主要缺口定位到 R3 旧类跨天保持：
旧类在 R3 占 80% 权重，但 Stage48 Old 只有约 0.23。本报告器只汇总
Stage67 的 seed7 训练结果，判断 replay-only old SupCon 是否真的提高
Old，同时不明显压塌 New 或 Overall。
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
}


def _r3_row(path: Path) -> pd.Series:
    """读取最终 R3 指标行。"""
    if not path.exists():
        raise FileNotFoundError(f"Missing result CSV: {path}")
    frame = pd.read_csv(path)
    if "Stage" in frame.columns:
        rows = frame.loc[frame["Stage"].astype(str) == "After R3"]
        return rows.iloc[-1] if not rows.empty else frame.tail(1).iloc[0]
    return frame.tail(1).iloc[0]


def _collect(root: Path, job_id: str, label: str, weight: float) -> dict[str, float | str]:
    """收集一个权重目录的 R3 指标和相对 Stage48 seed7 的变化。"""
    save_dir = root / f"lora_s67_oldreplay_supcon_{label}_seed7_{job_id}"
    row = _r3_row(save_dir / "incremental_results.csv")
    record: dict[str, float | str] = {
        "label": label,
        "weight": float(weight),
        "save_dir": str(save_dir),
        "overall": float(row["Overall Acc"]),
        "old": float(row["Old Acc"]),
        "new": float(row["New Acc"]),
        "forgetting": float(row["Forgetting Rate"]),
        "macro_f1": float(row["Macro F1"]),
    }
    recording_csv = save_dir / "recording_level_incremental_results.csv"
    if recording_csv.exists():
        recording = _r3_row(recording_csv)
        record["recording_overall"] = float(recording["Overall Acc"])
        record["recording_new"] = float(recording["New Acc"])
    for key, value in STAGE48_SEED7.items():
        record[f"delta_stage48_{key}"] = float(record[key]) - float(value)
    return record


def _passes(record: dict[str, float | str], baseline: dict[str, float | str]) -> bool:
    """预注册门槛：Old 明显改善，Overall 同步改善，New 不明显塌缩。"""
    if float(record["weight"]) <= 0:
        return False
    return (
        float(record["old"]) >= float(baseline["old"]) + 0.03
        and float(record["overall"]) >= float(baseline["overall"]) + 0.01
        and float(record["new"]) >= float(baseline["new"]) - 0.05
        and float(record["forgetting"]) <= float(baseline["forgetting"]) + 0.03
    )


def _fmt_optional(record: dict[str, float | str], key: str) -> str:
    """格式化可选 recording-level 指标。"""
    return f"{float(record[key]):.4f}" if key in record else "-"


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage67 LoRa old replay SupCon seed7 report")
    parser.add_argument("--root", default="results/stage67")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    variants = [("w0p00", 0.0), ("w0p05", 0.05), ("w0p10", 0.10)]
    records = [_collect(root, str(args.job_id), label, weight) for label, weight in variants]
    baseline = next(item for item in records if float(item["weight"]) == 0.0)
    best = max(records, key=lambda item: float(item["overall"]))
    gates = {str(item["label"]): _passes(item, baseline) for item in records}
    any_pass = any(gates.values())

    lines = [
        f"# Stage67 LoRa 旧类 replay SupCon Seed7 报告（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'有候选通过 seed7 门槛，可考虑扩三种子。' if any_pass else '未通过 seed7 门槛，不扩三种子。'}",
        f"- 同 job 权重 0 对照 R3 Overall/Old/New/Forgetting = "
        f"{float(baseline['overall']):.4f}/{float(baseline['old']):.4f}/{float(baseline['new']):.4f}/{float(baseline['forgetting']):.4f}。",
        f"- 最好 Overall：{best['label']}，R3 Overall/Old/New/Forgetting = "
        f"{float(best['overall']):.4f}/{float(best['old']):.4f}/{float(best['new']):.4f}/{float(best['forgetting']):.4f}。",
        "- 通过门槛要求：Old 比同 job 对照至少 +0.03、Overall 至少 +0.01、New 下降不超过 0.05、Forgetting 不恶化超过 0.03。",
        "",
        "## R3 指标",
        "",
        "| Variant | Weight | Overall | Old | New | Forgetting | ΔOverall vs Stage48 | ΔOld | ΔNew | ΔForgetting | Recording Overall | Recording New | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in records:
        label = str(item["label"])
        lines.append(
            "| {label} | {weight:.2f} | {overall:.4f} | {old:.4f} | {new:.4f} | {forgetting:.4f} | "
            "{delta_overall:+.4f} | {delta_old:+.4f} | {delta_new:+.4f} | {delta_forgetting:+.4f} | "
            "{recording_overall} | {recording_new} | {gate} |".format(
                label=label,
                weight=float(item["weight"]),
                overall=float(item["overall"]),
                old=float(item["old"]),
                new=float(item["new"]),
                forgetting=float(item["forgetting"]),
                delta_overall=float(item["delta_stage48_overall"]),
                delta_old=float(item["delta_stage48_old"]),
                delta_new=float(item["delta_stage48_new"]),
                delta_forgetting=float(item["delta_stage48_forgetting"]),
                recording_overall=_fmt_optional(item, "recording_overall"),
                recording_new=_fmt_optional(item, "recording_new"),
                gate="PASS" if gates[label] else "FAIL",
            )
        )
    lines.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage67_lora_old_replay_supcon_summary_v1",
                "job_id": str(args.job_id),
                "stage48_seed7": STAGE48_SEED7,
                "records": records,
                "baseline": baseline,
                "best": best,
                "gates": gates,
                "any_pass": bool(any_pass),
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
