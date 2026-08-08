"""汇总 Stage85 LoRa 新类-旧类 logit margin seed7 结果。

Stage85 只改变 RADCIL 训练期目标：对当前轮新伪类样本增加“新类伪标签
logit 高于旧类最大 logit”的 margin 损失。它不读取 held-out eval 真值，
也不改 discovery 前端。报告仍以 symbol-level R3 为正式判断口径。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


STAGE46_SEED7_BASELINE = {
    "overall": 0.3005,
    "old": 0.2429,
    "new": 0.5310,
    "forgetting": 0.1155,
}


def _read_r3_metrics(save_dir: Path) -> dict[str, float]:
    """读取主实验 incremental_results.csv 的 After R3 指标。"""

    csv_path = save_dir / "incremental_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing incremental results: {csv_path}")
    frame = pd.read_csv(csv_path)
    rows = frame[frame["Stage"].astype(str).str.strip().str.lower().isin({"after r3", "r3"})]
    row = rows.iloc[-1] if not rows.empty else frame.iloc[-1]
    return {
        "overall": float(row.get("Overall Acc", 0.0)),
        "old": float(row.get("Old Acc", 0.0)),
        "new": float(row.get("New Acc", 0.0)),
        "forgetting": float(row.get("Forgetting Rate", 0.0)),
        "macro_f1": float(row.get("Macro F1", 0.0)),
    }


def _read_recording_r3(save_dir: Path) -> dict[str, float] | None:
    """读取 recording-level 诊断指标；缺失时返回 None。"""

    csv_path = save_dir / "recording_level_incremental_results.csv"
    if not csv_path.exists():
        return None
    frame = pd.read_csv(csv_path)
    rows = frame[frame["Stage"].astype(str).str.strip().str.lower().isin({"after r3", "r3"})]
    row = rows.iloc[-1] if not rows.empty else frame.iloc[-1]
    return {
        "recording_overall": float(row.get("Overall Acc", 0.0)),
        "recording_new": float(row.get("New Acc", 0.0)),
    }


def main() -> int:
    """生成 Stage85 seed7 报告和 JSON 摘要。"""

    parser = argparse.ArgumentParser(description="Stage85 LoRa new-old margin seed7 报告器")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--margin-weight", type=float, required=True)
    parser.add_argument("--margin", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    metrics = _read_r3_metrics(save_dir)
    recording_metrics = _read_recording_r3(save_dir)
    deltas = {
        key: float(metrics[key]) - float(STAGE46_SEED7_BASELINE[key])
        for key in ("overall", "old", "new", "forgetting")
    }
    passed = (
        deltas["overall"] > 0.0
        and deltas["old"] >= -0.02
        and deltas["new"] >= -0.02
        and deltas["forgetting"] <= 0.05
    )
    recording_cells = (
        f"{recording_metrics['recording_overall']:.4f} | {recording_metrics['recording_new']:.4f}"
        if recording_metrics
        else "- | -"
    )
    lines = [
        f"# Stage85 LoRa 新类-旧类 logit margin seed7 报告（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'通过 seed7 门槛，可扩 seed13/31。' if passed else '未通过 seed7 门槛，暂不扩三种子。'}",
        f"- 配置：margin weight `{args.margin_weight}`，margin `{args.margin}`。",
        "- 改动范围：只在 RADCIL 训练期加入当前轮新伪类相对旧类最大 logit 的 margin 损失；strict split、GPCC recording-consensus、LoRa SSL 和 joint refinement 保持 Stage48 seed7 口径。",
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
            f"| stage85_new_old_margin | {metrics['overall']:.4f} | {metrics['old']:.4f} | "
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
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage85_lora_new_old_margin_seed7_summary_v1",
                "job_id": str(args.job_id),
                "save_dir": str(save_dir),
                "margin_weight": float(args.margin_weight),
                "margin": float(args.margin),
                "stage46_seed7_baseline": STAGE46_SEED7_BASELINE,
                "metrics": metrics,
                "recording_metrics": recording_metrics,
                "delta_vs_stage46_seed7": deltas,
                "passed_seed7_gate": bool(passed),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Stage85 报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
