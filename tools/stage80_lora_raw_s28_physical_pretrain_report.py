"""汇总 Stage80 LoRa 原始 I/Q s28 + 物理域预训练 seed7 结果。

Stage80 是对 Stage46 和 Stage38 的结构性合并：Stage46 已证明 Setup 1
原始 I/Q 重切到 1024 点、28 个 aligned symbol 后，LoRa 的 Overall/Old
会明显高于 Stage27 Chirp；Stage38 则验证过物理域自监督预训练。这里把
二者放在同一个 seed7 流程内比较，判断“更大原始数据 + 物理预训练”是否
比单独 raw-s28 更值得继续扩三种子。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# Stage46 raw-s28 seed7 的正式对照。Stage80 也是 seed7，所以先做同 seed
# 公平判断；若通过，再扩 seed13/31，而不是直接和三种子均值混在一起。
STAGE46_SEED7_BASELINE = {
    "overall": 0.3005,
    "old": 0.2429,
    "new": 0.5310,
    "forgetting": 0.1155,
}


def _read_r3_metrics(save_dir: Path) -> dict[str, float]:
    """读取 CIL 输出中的 After R3 指标。

    参数：
        save_dir: 单次实验输出目录，需包含 incremental_results.csv。

    返回：
        包含 Overall/Old/New/Forgetting/Macro F1 的浮点指标字典。
    """

    csv_path = save_dir / "incremental_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing incremental results: {csv_path}")
    frame = pd.read_csv(csv_path)
    rows = frame[frame["Stage"].astype(str).str.upper() == "AFTER R3"]
    row = rows.iloc[-1] if not rows.empty else frame.iloc[-1]
    return {
        "overall": float(row.get("Overall Acc", 0.0)),
        "old": float(row.get("Old Acc", 0.0)),
        "new": float(row.get("New Acc", 0.0)),
        "forgetting": float(row.get("Forgetting Rate", 0.0)),
        "macro_f1": float(row.get("Macro F1", 0.0)),
    }


def _read_recording_r3(save_dir: Path) -> dict[str, float] | None:
    """读取 recording-level 诊断指标；缺失时返回 None。

    该指标只作为 LoRa 多 symbol/transmission 诊断，不改变正式 symbol-level
    判断口径。
    """

    csv_path = save_dir / "recording_level_incremental_results.csv"
    if not csv_path.exists():
        return None
    frame = pd.read_csv(csv_path)
    rows = frame[frame["Stage"].astype(str).str.upper() == "AFTER R3"]
    row = rows.iloc[-1] if not rows.empty else frame.iloc[-1]
    return {
        "recording_overall": float(row.get("Overall Acc", 0.0)),
        "recording_new": float(row.get("New Acc", 0.0)),
    }


def main() -> int:
    """生成 Stage80 seed7 报告和 JSON 摘要。"""

    parser = argparse.ArgumentParser(description="Stage80 LoRa raw-s28 physical pretrain seed7 报告器")
    parser.add_argument("--root", default="results/stage80")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    save_dir = root / f"lora_raw_s28_physical_pretrain_seed7_{args.job_id}"
    checkpoint_summary_path = root / f"stage80_lora_raw_s28_physical_checkpoint_summary_{args.job_id}.json"
    metrics = _read_r3_metrics(save_dir)
    recording_metrics = _read_recording_r3(save_dir)
    pretrain_summary = json.loads(checkpoint_summary_path.read_text(encoding="utf-8"))

    deltas = {
        key: float(metrics[key]) - float(STAGE46_SEED7_BASELINE[key])
        for key in ("overall", "old", "new", "forgetting")
    }
    # 通过门槛强调“不能只救 Old、把 New 打塌”：Overall 至少要高于 Stage46
    # seed7，Old 不能掉，New 最多允许 2 个百分点以内的小回撤。
    passed = (
        deltas["overall"] > 0.0
        and deltas["old"] >= 0.0
        and deltas["new"] >= -0.02
        and deltas["forgetting"] <= 0.02
    )

    recording_cells = (
        f"{recording_metrics['recording_overall']:.4f} | {recording_metrics['recording_new']:.4f}"
        if recording_metrics
        else "- | -"
    )
    lines = [
        f"# Stage80 LoRa raw-s28 + 物理域预训练 seed7 报告（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'通过 seed7 门槛，可扩 seed13/31。' if passed else '未通过 seed7 门槛，暂不扩三种子。'}",
        f"- Checkpoint 选择：epoch `{pretrain_summary['best']['epoch']}`，IQ_7 val acc `{pretrain_summary['best']['validation_accuracy']:.4f}`。",
        "- 改动范围：只把 Stage46 raw-s28 初始 checkpoint 换成物理域预训练 checkpoint；strict split、GPCC、LoRa SSL、joint refinement 和 RADCIL 后端保持一致。",
        "",
        "## R3 指标",
        "",
        "| Variant | Overall | Old | New | Forgetting | Macro F1 | Recording Overall | Recording New |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| stage46_raw_s28_seed7 | {STAGE46_SEED7_BASELINE['overall']:.4f} | "
            f"{STAGE46_SEED7_BASELINE['old']:.4f} | {STAGE46_SEED7_BASELINE['new']:.4f} | "
            f"{STAGE46_SEED7_BASELINE['forgetting']:.4f} | n/a | 0.3067 | 0.6667 |"
        ),
        (
            f"| stage80_raw_s28_physical | {metrics['overall']:.4f} | {metrics['old']:.4f} | "
            f"{metrics['new']:.4f} | {metrics['forgetting']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{recording_cells} |"
        ),
        "",
        "## Delta vs Stage46 Seed7",
        "",
        f"- Overall: {deltas['overall']:+.4f}",
        f"- Old: {deltas['old']:+.4f}",
        f"- New: {deltas['new']:+.4f}",
        f"- Forgetting: {deltas['forgetting']:+.4f}",
        "",
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage80_lora_raw_s28_physical_pretrain_seed7_summary_v1",
                "job_id": str(args.job_id),
                "baseline": STAGE46_SEED7_BASELINE,
                "metrics": metrics,
                "recording_metrics": recording_metrics,
                "delta_vs_stage46_seed7": deltas,
                "pretrain": pretrain_summary,
                "passed_seed7_gate": bool(passed),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Stage80 报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
