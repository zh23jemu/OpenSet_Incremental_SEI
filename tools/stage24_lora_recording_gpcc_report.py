"""汇总 Stage 24 LoRa recording-level GPCC seed7 结果。

报告只读取 Slurm 生成的 CSV，持久化当前候选与既有 256 点严格基线的
差值。未知真实标签只出现在离线指标中，不参与 recording 聚合、聚类、
伪标签注册或模型训练。
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


def _read_r3_incremental(save_dir: Path) -> dict[str, float]:
    """读取 R3 增量核心指标。"""

    path = save_dir / "incremental_results.csv"
    frame = pd.read_csv(path)
    rows = frame[frame["Stage"].astype(str).str.contains("R3", case=False, na=False)]
    if rows.empty:
        raise RuntimeError(f"No R3 row found in {path}")
    row = rows.iloc[-1]
    return {
        "overall": float(row.get("Overall Acc", row.get("Overall", 0.0))),
        "old": float(row.get("Old Acc", row.get("Old", 0.0))),
        "new": float(row.get("New Acc", row.get("New", 0.0))),
        "forgetting": float(row.get("Forgetting Rate", row.get("Forgetting", 0.0))),
        "macro_f1": float(row.get("Macro F1", row.get("Macro F1 Score", 0.0))),
    }


def _read_clustering(save_dir: Path) -> list[dict[str, object]]:
    """读取各轮聚类诊断，确认 recording 聚合仍固定输出协议目标簇数。"""

    path = save_dir / "clustering_results.csv"
    frame = pd.read_csv(path)
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "round": str(row.get("Round", "")),
                "clusters": int(row.get("Final Cluster Count", 0)),
                "target": int(row.get("True New Classes", 0)),
                "hungarian": float(row.get("Hungarian Acc", 0.0)),
                "purity": float(row.get("Purity", 0.0)),
                "recording_groups": int(row.get("Recording-GPCC Groups", 0)),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 24 LoRa recording-level GPCC seed7 报告")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--job-id", default="manual")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    metrics = _read_r3_incremental(save_dir)
    clustering = _read_clustering(save_dir)
    deltas = {key: metrics[key] - BASELINE[key] for key in BASELINE}
    fixed_clusters = bool(clustering) and all(
        item["clusters"] == item["target"] == 5 for item in clustering
    )
    passed = bool(
        fixed_clusters
        and deltas["overall"] > 0.0
        and deltas["old"] >= 0.0
        and deltas["new"] >= -0.02
        and metrics["forgetting"] <= BASELINE["forgetting"]
    )
    summary = {
        "job_id": str(args.job_id),
        "save_dir": str(save_dir),
        "baseline": BASELINE,
        "r3": metrics,
        "delta": deltas,
        "clustering": clustering,
        "fixed_cluster_gate": fixed_clusters,
        "passed_seed7_gate": passed,
    }

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Stage 24 LoRa Recording-GPCC seed7 报告",
        "",
        f"- Job ID: `{args.job_id}`",
        "- 方法：先按可观测 `recording_id` 聚合同一次 transmission 的 symbol 特征，再用 GPCC 固定 5 簇并回填样本伪标签。",
        "- 边界：只使用 Day1 已知训练、Day2-4 当前 discovery 的 `recording_id` 与特征；IQ_8-10 held-out evaluation 不参与聚类或训练。",
        "",
        "| 指标 | Recording-GPCC R3 | 256 基线 | 差值 |",
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
    lines.extend(
        [
            "",
            "| 轮次 | 最终簇数 | 协议目标 | Hungarian | Purity | Recording 组数 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in clustering:
        lines.append(
            f"| {item['round']} | {item['clusters']} | {item['target']} | "
            f"{item['hungarian']:.4f} | {item['purity']:.4f} | {item['recording_groups']} |"
        )
    verdict = "通过 seed7 门槛，扩展 seed13/31。" if passed else "未通过 seed7 门槛，归档为负消融。"
    lines.extend(["", f"当前判定：{verdict}", ""])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Stage 24 报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
