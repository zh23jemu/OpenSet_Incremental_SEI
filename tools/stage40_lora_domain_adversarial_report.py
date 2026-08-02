"""汇总 Stage 40 LoRa 域对抗表征预训练 seed7 结果。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BASELINE_STAGE38 = {"overall": 0.2705, "old": 0.2131, "new": 0.5000, "forgetting": 0.2429}


def _r3(path: Path) -> dict[str, str]:
    """读取 CSV 的 AFTER R3 行。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"结果为空：{path}")
    for row in rows:
        if str(row.get("Stage", "")).strip().lower() == "after r3":
            return row
    return rows[-1]


def main() -> int:
    """生成 Stage 40 seed7 报告。"""

    parser = argparse.ArgumentParser(description="汇总 Stage 40 LoRa domain adversarial 结果")
    parser.add_argument("--root", default="results/stage40")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    save_dir = root / f"lora_domain_adversarial_seed7_{args.job_id}"
    symbol = _r3(save_dir / "incremental_results.csv")
    recording = _r3(save_dir / "recording_level_incremental_results.csv")
    checkpoint = json.loads(
        (root / f"stage40_lora_domain_adversarial_checkpoint_summary_{args.job_id}.json").read_text(
            encoding="utf-8"
        )
    )
    metrics = {
        "overall": float(symbol["Overall Acc"]),
        "old": float(symbol["Old Acc"]),
        "new": float(symbol["New Acc"]),
        "forgetting": float(symbol["Forgetting Rate"]),
        "macro_f1": float(symbol["Macro F1"]),
        "recording_overall": float(recording["Overall Acc"]),
        "recording_new": float(recording["New Acc"]),
    }
    passed = (
        metrics["overall"] > BASELINE_STAGE38["overall"]
        and metrics["old"] >= BASELINE_STAGE38["old"] - 0.01
        and metrics["new"] >= BASELINE_STAGE38["new"] - 0.02
    )
    verdict = "通过 seed7 门槛，可扩三种子。" if passed else "未通过 seed7 门槛，归档为负消融。"

    lines = [
        "# Stage 40 LoRa 域对抗表征预训练 Seed7 报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 方法：用 Day1 已知训练样本与 Day2-4 discovery 样本的 Day 域标签训练域头，并通过梯度反转约束 backbone 学习跨天不变特征。",
        "- 边界：Day2-4 只使用 IQ_1-7 discovery/enrollment；不读取任何 IQ_8-10 held-out eval，不使用未知设备类别标签。",
        f"- Checkpoint：epoch `{checkpoint['best']['epoch']}`，IQ_7 val acc `{checkpoint['best']['validation_accuracy']:.4f}`。",
        "",
        "| Variant | Overall | Old | New | Forgetting | Macro F1 | Recording Overall | Recording New |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Stage38 seed7 | {BASELINE_STAGE38['overall']:.4f} | {BASELINE_STAGE38['old']:.4f} | {BASELINE_STAGE38['new']:.4f} | {BASELINE_STAGE38['forgetting']:.4f} | n/a | n/a | n/a |",
        f"| Stage40 seed7 | {metrics['overall']:.4f} | {metrics['old']:.4f} | {metrics['new']:.4f} | {metrics['forgetting']:.4f} | {metrics['macro_f1']:.4f} | {metrics['recording_overall']:.4f} | {metrics['recording_new']:.4f} |",
        "",
        f"- 当前判定：{verdict}",
        "",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage40_lora_domain_adversarial_seed7_summary_v1",
                "job_id": str(args.job_id),
                "baseline": BASELINE_STAGE38,
                "metrics": metrics,
                "checkpoint": checkpoint,
                "passed": bool(passed),
                "verdict": verdict,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Stage 40 报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
