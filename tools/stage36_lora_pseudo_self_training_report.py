"""汇总 Stage 36 LoRa 伪标签自训练 seed7 结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VARIANTS = ("baseline", "proto0p10_cons0p10", "proto0p20_cons0p20")
BASELINE = {
    "overall": 0.2495,
    "old": 0.1929,
    "new": 0.4762,
    "forgetting": 0.2786,
}


def _read_r3(path: Path) -> dict[str, float]:
    """读取单个变体的 R3 指标，兼容历史 CSV 列名。"""

    frame = pd.read_csv(path / "incremental_results.csv")
    rows = frame[frame["Stage"].astype(str).str.upper() == "AFTER R3"]
    row = rows.iloc[-1] if not rows.empty else frame.iloc[-1]
    return {
        "overall": float(row.get("Overall Acc", row.get("Overall", 0.0))),
        "old": float(row.get("Old Acc", row.get("Old", 0.0))),
        "new": float(row.get("New Acc", row.get("New", 0.0))),
        "forgetting": float(row.get("Forgetting Rate", row.get("Forgetting", 0.0))),
        "macro_f1": float(row.get("Macro F1", row.get("Macro F1 Score", 0.0))),
    }


def main() -> int:
    """生成结果表，并按预注册门槛决定是否扩展种子。"""

    parser = argparse.ArgumentParser(description="汇总 Stage 36 LoRa 结果")
    parser.add_argument("--root", default="results/stage36")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    records = {}
    for variant in VARIANTS:
        save_dir = Path(args.root) / f"lora_pseudo_self_training_{variant}_seed7_{args.job_id}"
        records[variant] = _read_r3(save_dir)

    passed = []
    for variant, row in records.items():
        if (
            row["overall"] > BASELINE["overall"]
            and row["new"] >= 0.4952
            and row["old"] >= BASELINE["old"] - 0.02
        ):
            passed.append(variant)

    lines = [
        "# Stage 36 LoRa 伪标签自训练 seed7 报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 固定：Chirp、recording-consensus=0.65、cross-day、LoRa SSL、top0.80 注册。",
        "- 新增目标：当前轮伪类 prototype 归属损失 + 高置信伪标签增强一致性损失。",
        "",
        "| Variant | Overall | Old | New | Forgetting | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, row in records.items():
        lines.append(
            f"| {variant} | {row['overall']:.4f} | {row['old']:.4f} | "
            f"{row['new']:.4f} | {row['forgetting']:.4f} | {row['macro_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            (
                "- Stage 35 top0.80 对照："
                f"`{BASELINE['overall']:.4f}/{BASELINE['old']:.4f}/"
                f"{BASELINE['new']:.4f}/{BASELINE['forgetting']:.4f}`。"
            ),
            f"- 通过变体：`{', '.join(passed) if passed else '无'}`。",
            (
                "- 若无变体通过，说明静态伪标签自训练仍不足，下一步应转向 "
                "LoRa 物理域专用表征预训练或重新审查 recording-level 协议，"
                "不再继续相邻权重搜索。"
            ),
            "",
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage36_lora_pseudo_self_training_summary_v1",
                "job_id": str(args.job_id),
                "baseline": BASELINE,
                "records": records,
                "passed_variants": passed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Stage 36 报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
