"""汇总 Stage 39 LoRa discovery-unlabeled 物理预训练 seed7 结果。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE38_SEED7 = {"overall": 0.2705, "old": 0.2131, "new": 0.5000, "forgetting": 0.2429}
STAGE27_MULTI = {"overall": 0.2540, "old": 0.1952, "new": 0.4889, "forgetting": 0.2794}


def _read_rows(path: Path) -> list[dict[str, str]]:
    """读取结果 CSV，缺失时直接报错，避免报告器吞掉失败运行。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"结果为空：{path}")
    return rows


def _r3(path: Path) -> dict[str, str]:
    """优先取 AFTER R3 行，兼容只有最终行的旧 CSV。"""

    rows = _read_rows(path)
    for row in rows:
        if str(row.get("Stage", "")).strip().lower() == "after r3":
            return row
    return rows[-1]


def _metric(row: dict[str, str], key: str) -> float:
    """解析浮点指标。"""

    return float(row[key])


def main() -> int:
    """生成 Stage 39 seed7 报告。"""

    parser = argparse.ArgumentParser(description="汇总 Stage 39 LoRa discovery-unlabeled physical pretrain")
    parser.add_argument("--root", default="results/stage39")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    save_dir = root / f"lora_discovery_physical_seed7_{args.job_id}"
    symbol = _r3(save_dir / "incremental_results.csv")
    recording = _r3(save_dir / "recording_level_incremental_results.csv")
    ckpt_summary = json.loads(
        (root / f"stage39_lora_discovery_physical_checkpoint_summary_{args.job_id}.json").read_text(
            encoding="utf-8"
        )
    )
    metrics = {
        "overall": _metric(symbol, "Overall Acc"),
        "old": _metric(symbol, "Old Acc"),
        "new": _metric(symbol, "New Acc"),
        "forgetting": _metric(symbol, "Forgetting Rate"),
        "macro_f1": _metric(symbol, "Macro F1"),
        "recording_overall": _metric(recording, "Overall Acc"),
        "recording_new": _metric(recording, "New Acc"),
    }
    passed = (
        metrics["overall"] > STAGE38_SEED7["overall"]
        and metrics["old"] >= STAGE38_SEED7["old"] - 0.01
        and metrics["new"] >= STAGE38_SEED7["new"] - 0.02
    )
    verdict = "通过 seed7 门槛，可扩三种子。" if passed else "未通过 seed7 门槛，归档为负消融。"

    lines = [
        "# Stage 39 LoRa Discovery-Unlabeled 物理预训练 Seed7 报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 方法：在 Stage 38 物理预训练基础上，额外使用 Day2-4 IQ_1-7 discovery 样本做无标签物理描述符回归。",
        "- 边界：不读取 Day1 IQ_8-10 或 Day2-4 IQ_8-10 held-out eval，不使用未知设备标签。",
        f"- Checkpoint：epoch `{ckpt_summary['best']['epoch']}`，IQ_7 val acc `{ckpt_summary['best']['validation_accuracy']:.4f}`，无标签 discovery 样本 `{ckpt_summary['unlabeled_discovery_samples']}`。",
        "",
        "| Variant | Overall | Old | New | Forgetting | Macro F1 | Recording Overall | Recording New |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Stage27 三种子候选 | {STAGE27_MULTI['overall']:.4f} | {STAGE27_MULTI['old']:.4f} | {STAGE27_MULTI['new']:.4f} | {STAGE27_MULTI['forgetting']:.4f} | n/a | n/a | n/a |",
        f"| Stage38 seed7 | {STAGE38_SEED7['overall']:.4f} | {STAGE38_SEED7['old']:.4f} | {STAGE38_SEED7['new']:.4f} | {STAGE38_SEED7['forgetting']:.4f} | n/a | n/a | n/a |",
        f"| Stage39 seed7 | {metrics['overall']:.4f} | {metrics['old']:.4f} | {metrics['new']:.4f} | {metrics['forgetting']:.4f} | {metrics['macro_f1']:.4f} | {metrics['recording_overall']:.4f} | {metrics['recording_new']:.4f} |",
        "",
        f"- 当前判定：{verdict}",
        "",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "schema_version": "stage39_lora_discovery_physical_seed7_summary_v1",
        "job_id": str(args.job_id),
        "baseline_stage38_seed7": STAGE38_SEED7,
        "baseline_stage27_multiseed": STAGE27_MULTI,
        "metrics": metrics,
        "checkpoint": ckpt_summary,
        "passed": bool(passed),
        "verdict": verdict,
    }
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Stage 39 报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
