"""汇总 Stage 23 LoRa 长窗 seed7 验证结果。

该报告器只读取 Slurm 已生成的小型 CSV/JSON 文件，不读取 checkpoint、
replay memory 或 held-out evaluation 原始 IQ。它的作用是把长窗 LoRa
候选和当前 256 点基线放在同一口径下，便于快速判断是否值得扩三种子。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASELINE = {
    "overall": 0.1667,
    "old": 0.0917,
    "new": 0.4667,
    "forgetting": 0.4857,
}


def _read_r3_metrics(save_dir: Path) -> dict[str, float]:
    """读取完整增量结果中的 R3 行，并返回客户关心的四个核心指标。"""

    path = save_dir / "incremental_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing incremental results: {path}")
    frame = pd.read_csv(path)
    rows = frame[frame["Stage"].astype(str).str.contains("R3", case=False, na=False)]
    if rows.empty:
        raise RuntimeError(f"No R3 row found in {path}")
    row = rows.iloc[-1]
    return {
        "overall": float(row.get("Overall Acc", row.get("Overall", 0.0))),
        "old": float(row.get("Old Acc", row.get("Old", 0.0))),
        "new": float(row.get("New Acc", row.get("New", 0.0))),
        "forgetting": float(row.get("Forgetting", 0.0)),
        "macro_f1": float(row.get("Macro F1", row.get("Macro F1 Score", 0.0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 23 LoRa 长窗 seed7 报告器")
    parser.add_argument("--save-dir", required=True, help="单个 seed7 实验输出目录")
    parser.add_argument("--output", required=True, help="Markdown 报告输出路径")
    parser.add_argument("--summary-json", required=True, help="JSON 摘要输出路径")
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--sample-length", type=int, default=1024)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    metrics = _read_r3_metrics(save_dir)
    deltas = {key: metrics[key] - BASELINE[key] for key in BASELINE}
    passed = (
        deltas["overall"] > 0.0
        and deltas["old"] >= 0.0
        and deltas["new"] >= -0.02
        and metrics["forgetting"] <= BASELINE["forgetting"]
    )

    summary = {
        "job_id": str(args.job_id),
        "save_dir": str(save_dir),
        "sample_length": int(args.sample_length),
        "baseline": BASELINE,
        "r3": metrics,
        "delta": deltas,
        "passed_seed7_gate": bool(passed),
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    verdict = (
        "通过 seed7 门槛，建议扩展 seed13/31。"
        if passed
        else "未通过 seed7 门槛，不扩三种子。"
    )
    lines = [
        "# Stage 23 LoRa 长窗 seed7 报告",
        "",
        f"- Job ID: `{args.job_id}`",
        f"- 样本长度: `{int(args.sample_length)}`",
        f"- 输出目录: `{save_dir}`",
        "",
        "| 指标 | 长窗 R3 | 256 基线 | 差值 |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("overall", "Overall"),
        ("old", "Old"),
        ("new", "New"),
        ("forgetting", "Forgetting"),
    ):
        lines.append(
            f"| {label} | {metrics[key]:.4f} | {BASELINE[key]:.4f} | {deltas[key]:+.4f} |"
        )
    lines.extend(["", f"当前判定：{verdict}", ""])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
