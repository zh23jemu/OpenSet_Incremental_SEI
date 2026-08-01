"""汇总 Stage 35 LoRa 联合注册/训练 seed7 结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VARIANTS = ("registration_all", "registration_top0p60", "registration_top0p80")
BASELINE = {"overall": 0.2476, "old": 0.1857, "new": 0.4952, "forgetting": 0.2690}


def _r3(path: Path) -> dict[str, float]:
    """读取单个变体的 After R3 指标。"""

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
    """生成结果表，并判断是否值得扩展三种子。"""

    parser = argparse.ArgumentParser(description="汇总 Stage 35 LoRa 结果")
    parser.add_argument("--root", default="results/stage35")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    records = {}
    for variant in VARIANTS:
        save_dir = Path(args.root) / f"lora_joint_registration_{variant}_seed7_{args.job_id}"
        records[variant] = _r3(save_dir)

    passed = []
    for variant, row in records.items():
        if (
            row["overall"] > BASELINE["overall"]
            and not (
                row["old"] < BASELINE["old"] and row["new"] < BASELINE["new"]
            )
        ):
            passed.append(variant)

    lines = [
        "# Stage 35 LoRa 联合注册/训练 seed7 报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 固定：Chirp、recording-consensus=0.65、cross-day、LoRa SSL、joint discovery-CIL。",
        "- 结构：高置信样本只用于新类 imprint/replay；全量 discovery 仍参与带权当前轮训练。",
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
            f"- Stage 34 对照：`{BASELINE['overall']:.4f}/{BASELINE['old']:.4f}/{BASELINE['new']:.4f}/{BASELINE['forgetting']:.4f}`。",
            f"- 通过变体：`{', '.join(passed) if passed else '无'}`。",
            "- 若无变体通过，停止注册比例搜索，转向真正的联合伪标签自训练目标。",
            "",
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage35_lora_joint_registration_summary_v1",
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
    print(f"Stage 35 报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
