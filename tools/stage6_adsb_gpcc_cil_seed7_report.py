"""生成阶段 6 ADS-B GPCC + Long-RADCIL seed7 完整增量报告。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _float(row: pd.Series, key: str, default: float = float("nan")) -> float:
    """从 pandas 行中安全读取浮点指标。"""

    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _fmt(value: Any) -> str:
    """四位小数输出，便于和阶段 4 报告直接对照。"""

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def collect_metrics(save_dir: Path) -> dict[str, Any]:
    """读取完整增量输出，抽取 R1/R2/R3 前端与后端指标。"""

    clustering = pd.read_csv(save_dir / "clustering_results.csv")
    incremental = pd.read_csv(save_dir / "incremental_results.csv")
    per_round_path = save_dir / "per_round_summary_results.csv"
    per_round = pd.read_csv(per_round_path) if per_round_path.exists() else pd.DataFrame()

    rounds: list[dict[str, Any]] = []
    for _, row in clustering.iterrows():
        round_name = str(row.get("Round", ""))
        inc_match = incremental[incremental["Stage"].astype(str) == f"After {round_name}"]
        inc_row = inc_match.iloc[0] if not inc_match.empty else pd.Series(dtype=object)
        rounds.append({
            "round": round_name,
            "cluster_count": int(_float(row, "Final Cluster Count", 0.0)),
            "target_classes": int(_float(row, "True New Classes", 0.0)),
            "cluster_error": int(_float(row, "Cluster Count Error", 0.0)),
            "ari": _float(row, "ARI"),
            "hungarian_acc": _float(row, "Hungarian Acc"),
            "overall_acc": _float(inc_row, "Overall Acc"),
            "old_acc": _float(inc_row, "Old Acc"),
            "new_acc": _float(inc_row, "New Acc"),
            "forgetting": _float(inc_row, "Forgetting Rate"),
            "macro_f1": _float(inc_row, "Macro F1"),
        })

    r3 = next((row for row in rounds if row["round"] == "R3"), rounds[-1] if rounds else {})
    return {
        "save_dir": str(save_dir),
        "rounds": rounds,
        "r3": r3,
        "per_round_rows": per_round.to_dict("records") if not per_round.empty else [],
    }


def build_markdown(job_id: str, metrics: dict[str, Any]) -> str:
    """构造简洁报告，突出是否超过 ADS-B 当前约 0.49 的 R3 Overall。"""

    r3 = metrics.get("r3", {})
    lines = [
        "# 阶段 6 ADS-B GPCC + Long-RADCIL seed7 完整增量报告",
        "",
        f"- 生成时间 UTC：{datetime.now(timezone.utc).isoformat()}",
        f"- Slurm Job：{job_id}",
        f"- 输出目录：`{metrics['save_dir']}`",
        "- 前端：GPCC，固定每轮 10 簇，不调用 HDBSCAN。",
        "- 后端：当前 ADS-B Long-RADCIL 参数，未启用 target split。",
        "",
        "## 结论",
        "",
    ]
    r3_overall = float(r3.get("overall_acc", float("nan")))
    if r3_overall > 0.4932:
        lines.append(f"- R3 Overall `{_fmt(r3_overall)}`，超过默认 target split 后端对照约 `0.4932`，可进入三种子验证。")
    else:
        lines.append(f"- R3 Overall `{_fmt(r3_overall)}`，暂未超过默认 target split 后端对照约 `0.4932`，先不扩三种子。")
    lines.extend(["", "## 分轮指标", ""])
    lines.append("| Round | K | Target | Error | ARI | Hungarian | Overall | Old | New | Forgetting | Macro F1 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in metrics["rounds"]:
        lines.append(
            "| {round} | {cluster_count} | {target_classes} | {cluster_error} | {ari} | {hungarian} | {overall} | {old} | {new} | {forgetting} | {macro_f1} |".format(
                round=row["round"],
                cluster_count=row["cluster_count"],
                target_classes=row["target_classes"],
                cluster_error=row["cluster_error"],
                ari=_fmt(row["ari"]),
                hungarian=_fmt(row["hungarian_acc"]),
                overall=_fmt(row["overall_acc"]),
                old=_fmt(row["old_acc"]),
                new=_fmt(row["new_acc"]),
                forgetting=_fmt(row["forgetting"]),
                macro_f1=_fmt(row["macro_f1"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 ADS-B GPCC + Long-RADCIL seed7 报告。")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    metrics = collect_metrics(Path(args.save_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(args.job_id, metrics), encoding="utf-8")

    summary = Path(args.summary_json)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps({"job_id": args.job_id, "metrics": metrics}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"ADS-B GPCC CIL seed7 report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
