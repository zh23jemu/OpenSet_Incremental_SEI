#!/usr/bin/env python3
"""Stage88：复核 Stage87 support 诊断的原始 R3 类比例口径。

Stage87 把同日旧类 support 样本从评估分母中剔除后，R3 Overall 达到 0.5133。
但 R3 原始协议是 20 个旧类 + 5 个新类，旧类权重约 80%。support 剔除后，
剩余评估集的旧/新比例会变化，因此需要额外报告“按原始 20:5 类比例重加权”
后的 Overall，避免把分母变化误写成模型能力提升。

本脚本只读取 Stage87 summary，不读任何额外数据，不改变实验结果。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    """统一输出 4 位小数。"""

    return f"{float(value):.4f}"


def _adjusted_overall(row: dict[str, Any], old_classes: int, new_classes: int) -> float:
    """按原始 seen old/new 类数量重新计算类比例加权 Overall。"""

    old_weight = float(old_classes) / float(old_classes + new_classes)
    new_weight = float(new_classes) / float(old_classes + new_classes)
    return old_weight * float(row["old"]) + new_weight * float(row["new"])


def _required_old_for_target(target: float, new_acc: float, old_classes: int, new_classes: int) -> float:
    """在固定 New Acc 时，达到目标 Overall 所需 Old Acc。"""

    old_weight = float(old_classes) / float(old_classes + new_classes)
    new_weight = float(new_classes) / float(old_classes + new_classes)
    return (float(target) - new_weight * float(new_acc)) / old_weight


def main() -> int:
    """生成 Stage88 原始类比例复核报告。"""

    parser = argparse.ArgumentParser(description="Stage88 LoRa support adjusted-mix report")
    parser.add_argument("--stage87-summary", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--old-classes", type=int, default=20)
    parser.add_argument("--new-classes", type=int, default=5)
    parser.add_argument("--target-overall", type=float, default=0.50)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    summary_path = Path(args.stage87_summary)
    stage87 = json.loads(summary_path.read_text(encoding="utf-8"))
    adjusted_rows: list[dict[str, Any]] = []
    for result in stage87["results"]:
        for row in result["r3_rows"]:
            adjusted = _adjusted_overall(row, args.old_classes, args.new_classes)
            required_old = _required_old_for_target(
                args.target_overall,
                new_acc=float(row["new"]),
                old_classes=args.old_classes,
                new_classes=args.new_classes,
            )
            adjusted_rows.append(
                {
                    "support_recordings_per_old_class": int(row["support_recordings_per_old_class"]),
                    "variant": row["variant"],
                    "observed_support_excluded_overall": float(row["overall"]),
                    "old": float(row["old"]),
                    "new": float(row["new"]),
                    "forgetting": float(row["forgetting"]),
                    "adjusted_original_mix_overall": float(adjusted),
                    "required_old_for_target": float(required_old),
                    "old_gap_to_target": float(row["old"]) - float(required_old),
                    "support_samples": int(row["support_samples"]),
                    "eval_samples": int(row["eval_samples"]),
                    "passes_observed_50": float(row["overall"]) >= float(args.target_overall),
                    "passes_adjusted_50": float(adjusted) >= float(args.target_overall),
                }
            )
    best_adjusted = max(
        adjusted_rows,
        key=lambda row: (
            float(row["adjusted_original_mix_overall"]),
            float(row["old"]),
            float(row["new"]),
        ),
    )
    observed_best = stage87["best_r3"]
    observed_best_adjusted = next(
        row
        for row in adjusted_rows
        if row["support_recordings_per_old_class"] == int(observed_best["support_recordings_per_old_class"])
        and row["variant"] == observed_best["variant"]
    )

    lines = [
        f"# Stage88 LoRa support 原始类比例复核（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        (
            "- 当前判定："
            + (
                "按原始 20旧/5新类比例重加权后仍达到 50%。"
                if bool(best_adjusted["passes_adjusted_50"])
                else "Stage87 的 50% 主要受 support 剔除后的分母变化影响；按原始 20旧/5新类比例重算后未过 50%。"
            )
        ),
        (
            f"- Stage87 observed best：`{observed_best['variant']}`，support/old-class="
            f"`{observed_best['support_recordings_per_old_class']}`，observed Overall `{_fmt(observed_best['overall'])}`；"
            f"按原始类比例重算 Overall `{_fmt(observed_best_adjusted['adjusted_original_mix_overall'])}`。"
        ),
        (
            f"- 最佳 adjusted 结果：`{best_adjusted['variant']}`，support/old-class="
            f"`{best_adjusted['support_recordings_per_old_class']}`，Adjusted Overall/Old/New="
            f"`{_fmt(best_adjusted['adjusted_original_mix_overall'])}/{_fmt(best_adjusted['old'])}/{_fmt(best_adjusted['new'])}`。"
        ),
        (
            f"- 若 New 固定为 `{_fmt(best_adjusted['new'])}`，要达到 50% 原始类比例 Overall，"
            f"Old 需 `{_fmt(best_adjusted['required_old_for_target'])}`；当前 Old 差值 `{_fmt(best_adjusted['old_gap_to_target'])}`。"
        ),
        "",
        "## R3 复核表",
        "",
        "| Support / Old Class | Variant | Observed Overall | Adjusted Overall | Old | New | Required Old @50 | Old Gap | Pass Observed | Pass Adjusted |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in adjusted_rows:
        lines.append(
            "| {support} | {variant} | {observed} | {adjusted} | {old} | {new} | {required_old} | {old_gap} | {pass_observed} | {pass_adjusted} |".format(
                support=int(row["support_recordings_per_old_class"]),
                variant=row["variant"],
                observed=_fmt(row["observed_support_excluded_overall"]),
                adjusted=_fmt(row["adjusted_original_mix_overall"]),
                old=_fmt(row["old"]),
                new=_fmt(row["new"]),
                required_old=_fmt(row["required_old_for_target"]),
                old_gap=_fmt(row["old_gap_to_target"]),
                pass_observed=str(bool(row["passes_observed_50"])),
                pass_adjusted=str(bool(row["passes_adjusted_50"])),
            )
        )
    lines.extend(
        [
            "",
            "## 口径说明",
            "",
            "- Observed Overall 是 Stage87 support 样本剔除后的实际剩余 held-out 分母。",
            "- Adjusted Overall 按 R3 原始 20 个旧类、5 个新类重新加权，避免 support 剔除改变旧/新比例。",
            "- 该报告仍是 support 数据需求诊断，不是无 support strict 正式成绩。",
            "",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage88_lora_support_adjusted_mix_v1",
                "job_id": str(args.job_id),
                "stage87_summary": str(summary_path),
                "old_classes": int(args.old_classes),
                "new_classes": int(args.new_classes),
                "target_overall": float(args.target_overall),
                "adjusted_rows": adjusted_rows,
                "stage87_observed_best": observed_best,
                "stage87_observed_best_adjusted": observed_best_adjusted,
                "best_adjusted_r3": best_adjusted,
                "passes_adjusted_50": bool(best_adjusted["passes_adjusted_50"]),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Stage88 报告：{output}")
    print(json.dumps({"best_adjusted_r3": best_adjusted}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
