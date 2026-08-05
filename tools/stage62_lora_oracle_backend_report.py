#!/usr/bin/env python3
"""汇总 Stage62 LoRa oracle discovery 下的后端保护诊断。

Stage62 固定 oracle discovery，只比较 RADCIL base、old-prototype-route 与
grouped DOI-style fusion。该实验读取 discovery 真值，仍然只能作为上界诊断。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


STAGE61_ORACLE_BASE = {
    "overall": 0.2986,
    "old": 0.2220,
    "new": 0.6048,
    "forgetting": 0.2095,
}


def _r3(path: Path) -> pd.Series:
    """读取 R3 指标行。"""
    frame = pd.read_csv(path)
    row = frame.loc[frame["Stage"].astype(str) == "After R3"]
    if row.empty:
        row = frame.tail(1)
    return row.iloc[-1]


def collect(root: Path, job_id: str, variant: str) -> dict[str, float | str]:
    """收集一个变体的 R3 指标与可选校准诊断。"""
    save_dir = root / f"lora_s62_{variant}_{job_id}"
    row = _r3(save_dir / "incremental_results.csv")
    record: dict[str, float | str] = {
        "variant": variant,
        "save_dir": str(save_dir),
        "overall": float(row["Overall Acc"]),
        "old": float(row["Old Acc"]),
        "new": float(row["New Acc"]),
        "forgetting": float(row["Forgetting Rate"]),
        "macro_f1": float(row["Macro F1"]),
    }
    for name, column, out_key in [
        ("old_prototype_route_iq7_calibration.csv", "Old Mass Threshold", "old_route_threshold"),
        ("grouped_doi_iq7_calibration.csv", "Fusion Weight", "grouped_doi_weight"),
    ]:
        path = save_dir / name
        if path.exists():
            calibration = pd.read_csv(path)
            if not calibration.empty and column in calibration.columns:
                record[out_key] = float(calibration.iloc[-1][column])
    return record


def passes_gate(record: dict[str, float | str], base: dict[str, float | str]) -> bool:
    """诊断门槛：在同一个 job 的 oracle base 上，同时保住 New 并明显救 Old/Overall。"""
    if record["variant"] == "oracle_base":
        return False
    return (
        float(record["overall"]) >= float(base["overall"]) + 0.03
        and float(record["old"]) >= float(base["old"]) + 0.03
        and float(record["new"]) >= float(base["new"]) - 0.05
    )


def fmt_optional(record: dict[str, float | str], key: str) -> str:
    """格式化可选校准字段。"""
    return f"{float(record[key]):.4f}" if key in record else "-"


def table(records: list[dict[str, float | str]], base: dict[str, float | str]) -> str:
    """生成 Markdown 表格。"""
    lines = [
        "| Variant | Overall | Old | New | Forgetting | ΔOverall vs oracle base | ΔOld | ΔNew | Route Thr | DOI Weight | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in records:
        lines.append(
            "| {variant} | {overall:.4f} | {old:.4f} | {new:.4f} | {forgetting:.4f} | {do:+.4f} | {dold:+.4f} | {dn:+.4f} | {thr} | {weight} | {gate} |".format(
                variant=item["variant"],
                overall=float(item["overall"]),
                old=float(item["old"]),
                new=float(item["new"]),
                forgetting=float(item["forgetting"]),
                do=float(item["overall"]) - float(base["overall"]),
                dold=float(item["old"]) - float(base["old"]),
                dn=float(item["new"]) - float(base["new"]),
                thr=fmt_optional(item, "old_route_threshold"),
                weight=fmt_optional(item, "grouped_doi_weight"),
                gate="PASS" if passes_gate(item, base) else "FAIL",
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage62 LoRa oracle discovery 后端诊断报告器")
    parser.add_argument("--root", default="results/stage62")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--variants", nargs="+", default=["oracle_base", "oracle_oldroute", "oracle_grouped_doi"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    records = [collect(root, str(args.job_id), variant) for variant in args.variants]
    base = next((item for item in records if item["variant"] == "oracle_base"), records[0])
    passed = [str(item["variant"]) for item in records if passes_gate(item, base)]
    best = max(records, key=lambda item: float(item["overall"]))
    lines = [
        f"# Stage62 LoRa Oracle Discovery 后端诊断（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'存在 oracle 后端上界候选。' if passed else '未发现 Old/New 同时可行的后端上界。'}",
        f"- 最佳 Overall：{best['variant']}，R3 Overall={float(best['overall']):.4f}，Old={float(best['old']):.4f}，New={float(best['new']):.4f}。",
        f"- 本次 oracle base：R3 Overall={float(base['overall']):.4f}，Old={float(base['old']):.4f}，New={float(base['new']):.4f}。",
        "- 注意：所有变体都使用 oracle discovery，结果只能做瓶颈诊断，不能作为正式方法。",
        f"- Stage61 参考 oracle base：Overall={STAGE61_ORACLE_BASE['overall']:.4f}，Old={STAGE61_ORACLE_BASE['old']:.4f}，New={STAGE61_ORACLE_BASE['new']:.4f}。",
        "",
        "## R3 指标",
        "",
        table(records, base),
        "",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage62_lora_oracle_backend_summary_v1",
                "job_id": str(args.job_id),
                "oracle_warning": "uses current discovery true labels; diagnostic upper bound only",
                "stage61_oracle_base": STAGE61_ORACLE_BASE,
                "job_oracle_base": base,
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
