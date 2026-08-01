"""汇总 Stage 38 LoRa 物理域自监督预训练 seed7 结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASELINE = {"overall": 0.2552, "old": 0.1988, "new": 0.4810, "forgetting": 0.2786}


def _read_r3(path: Path) -> dict[str, float]:
    """读取 CIL 结果中的 R3 指标。"""

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
    """生成 Stage 38 报告。"""

    parser = argparse.ArgumentParser(description="汇总 Stage 38 LoRa physical pretrain 结果")
    parser.add_argument("--root", default="results/stage38")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    save_dir = root / f"lora_physical_pretrain_seed7_{args.job_id}"
    metrics = _read_r3(save_dir)
    pretrain_summary_path = root / f"stage38_lora_physical_pretrain_checkpoint_summary_{args.job_id}.json"
    pretrain_summary = json.loads(pretrain_summary_path.read_text(encoding="utf-8"))
    passed = (
        metrics["overall"] > BASELINE["overall"]
        and metrics["new"] >= BASELINE["new"] - 0.01
        and metrics["old"] >= BASELINE["old"] - 0.02
    )
    lines = [
        "# Stage 38 LoRa 物理域自监督预训练 Seed7 报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        f"- Checkpoint 选择：epoch `{pretrain_summary['best']['epoch']}`，IQ_7 val acc `{pretrain_summary['best']['validation_accuracy']:.4f}`。",
        "- 改动范围：只替换初始 LoRa Chirp checkpoint；GPCC、cross-day、LoRa SSL、top0.80 注册和 RADCIL 后端保持 Stage 37 baseline 配置。",
        "",
        "| Variant | Overall | Old | New | Forgetting | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| stage37_baseline | {BASELINE['overall']:.4f} | {BASELINE['old']:.4f} | {BASELINE['new']:.4f} | {BASELINE['forgetting']:.4f} | n/a |",
        f"| physical_pretrain | {metrics['overall']:.4f} | {metrics['old']:.4f} | {metrics['new']:.4f} | {metrics['forgetting']:.4f} | {metrics['macro_f1']:.4f} |",
        "",
        f"- 是否通过 seed7 门槛：`{'是' if passed else '否'}`。",
        "- 若未通过，LoRa 低分进入局限收口，不再继续同类小矩阵。",
        "",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage38_lora_physical_pretrain_summary_v1",
                "job_id": str(args.job_id),
                "baseline": BASELINE,
                "metrics": metrics,
                "pretrain": pretrain_summary,
                "passed": bool(passed),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Stage 38 报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
