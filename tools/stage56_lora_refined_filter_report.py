#!/usr/bin/env python3
"""汇总 Stage56 LoRa joint refinement 高置信一致过滤 seed7 结果。

Stage56 固定 Stage48 的 raw s28 + recording-consensus 0.65 主配置，只在
joint discovery refinement 后的第二次 CIL 前过滤训练样本。报告器只读取
训练生成的 CSV/JSON，不参与训练、不选参、不读取未知真值。
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
    """读取最终 R3 行，兼容 incremental 与 clustering 两类输出格式。"""
    if not path.exists():
        raise FileNotFoundError(f"Missing result CSV: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty result CSV: {path}")
    if "Round" in frame.columns:
        row = frame.loc[frame["Round"].astype(str).isin({"3", "R3", "R3_refined"})]
        if row.empty:
            row = frame.tail(1)
    elif "Stage" in frame.columns:
        normalized = frame["Stage"].astype(str).str.strip().str.lower()
        row = frame.loc[normalized.isin({"after r3", "r3"})]
    else:
        raise KeyError(f"{path} has neither Round nor Stage column: {list(frame.columns)}")
    if row.empty:
        raise ValueError(f"Missing R3 row in {path}")
    return row.iloc[-1]


def _metric(row: pd.Series, name: str, default: float = 0.0) -> float:
    """把 pandas 标量转为普通 float；缺失诊断列时返回默认值。"""
    if name not in row.index or pd.isna(row[name]):
        return float(default)
    return float(row[name])


def collect_variant(root: Path, variant: str, job_id: str) -> dict[str, float | str]:
    """收集一个 Stage56 变体的 symbol、recording 和 refined filter 诊断。"""
    save_dir = root / f"lora_s56_{variant}_{job_id}"
    symbol = _r3_row(save_dir / "incremental_results.csv")
    record: dict[str, float | str] = {
        "variant": variant,
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
    refined_csv = save_dir / "joint_discovery_refinement_results.csv"
    if refined_csv.exists():
        refined = pd.read_csv(refined_csv)
        if not refined.empty:
            for key in (
                "Refined Filter Kept",
                "Refined Filter Total",
                "Refined Filter Keep Rate",
                "Refined Filter Agreement Rate",
                "Refined Filter Mean Confidence",
                "Refined Filter Fallback Classes",
            ):
                if key in refined.columns:
                    record[f"mean_{key.lower().replace(' ', '_')}"] = float(refined[key].mean())
    return record


def add_deltas(records: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    """补充相对本 job base 与 Stage48 seed7 的差值。"""
    base = next((item for item in records if item["variant"] == "base"), records[0])
    enriched: list[dict[str, float | str]] = []
    for item in records:
        row = dict(item)
        for key in ("overall", "old", "new", "forgetting"):
            row[f"delta_base_{key}"] = float(item[key]) - float(base[key])
            row[f"delta_stage48_{key}"] = float(item[key]) - float(STAGE48_SEED7[key])
        enriched.append(row)
    return enriched


def passes_gate(item: dict[str, float | str]) -> bool:
    """预注册门槛：必须提高 Overall，并保住 Old/New/Forgetting。"""
    if item["variant"] == "base":
        return False
    return (
        float(item["overall"]) >= STAGE48_SEED7["overall"] + 0.01
        and float(item["old"]) >= STAGE48_SEED7["old"]
        and float(item["new"]) >= STAGE48_SEED7["new"] - 0.02
        and float(item["forgetting"]) <= STAGE48_SEED7["forgetting"] + 0.02
    )


def fmt_optional(record: dict[str, float | str], key: str) -> str:
    """格式化可选诊断指标。"""
    return f"{float(record[key]):.4f}" if key in record else "-"


def table_md(records: list[dict[str, float | str]]) -> str:
    """生成精简 Markdown 表格，便于直接粘到客户/项目汇报。"""
    lines = [
        "| Variant | Overall | Old | New | Forgetting | ΔOverall vs Stage48 | ΔNew vs Stage48 | Refined Keep | Refined Agree | Recording Overall | Recording New | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in records:
        lines.append(
            "| {variant} | {overall:.4f} | {old:.4f} | {new:.4f} | {forgetting:.4f} | "
            "{delta_overall:+.4f} | {delta_new:+.4f} | {keep} | {agree} | {rec_overall} | {rec_new} | {gate} |".format(
                variant=item["variant"],
                overall=float(item["overall"]),
                old=float(item["old"]),
                new=float(item["new"]),
                forgetting=float(item["forgetting"]),
                delta_overall=float(item["delta_stage48_overall"]),
                delta_new=float(item["delta_stage48_new"]),
                keep=fmt_optional(item, "mean_refined_filter_keep_rate"),
                agree=fmt_optional(item, "mean_refined_filter_agreement_rate"),
                rec_overall=fmt_optional(item, "recording_overall"),
                rec_new=fmt_optional(item, "recording_new"),
                gate="PASS" if passes_gate(item) else "FAIL",
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage56 LoRa refined filter seed7 报告器")
    parser.add_argument("--root", default="results/stage56")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--variants", nargs="+", default=["base", "filter_top060", "filter_top080"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    records = add_deltas([collect_variant(root, variant, str(args.job_id)) for variant in args.variants])
    passed = [str(item["variant"]) for item in records if passes_gate(item)]
    best_overall = max(records, key=lambda item: float(item["overall"]))
    best_new = max(records, key=lambda item: float(item["new"]))

    lines = [
        f"# Stage56 LoRa Refined Filter Seed7 报告（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'通过 seed7 门槛，可进入三种子。' if passed else '未通过 seed7 门槛，不扩三种子。'}",
        f"- 最佳 Overall：{best_overall['variant']}，R3 Overall={float(best_overall['overall']):.4f}，New={float(best_overall['new']):.4f}",
        f"- 最佳 New：{best_new['variant']}，R3 New={float(best_new['new']):.4f}，Overall={float(best_new['overall']):.4f}",
        "",
        "## R3 指标",
        "",
        table_md(records),
        "",
        "## 判定规则",
        "",
        "- 固定 Stage48 raw s28 + recording-consensus 0.65 + Chirp + joint discovery-CIL 配置。",
        "- 只改变第二次 joint refinement CIL 的当前轮训练样本，不使用 held-out eval 真值或未知类真值选参。",
        "- 通过门槛：R3 Overall 至少比 Stage48 seed7 高 0.01，Old 不下降，New 下降不超过 0.02，Forgetting 恶化不超过 0.02。",
        "",
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage56_lora_refined_filter_summary_v1",
                "job_id": str(args.job_id),
                "stage48_seed7": STAGE48_SEED7,
                "records": records,
                "passed_variants": passed,
                "gate": "PASS" if passed else "FAIL",
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
