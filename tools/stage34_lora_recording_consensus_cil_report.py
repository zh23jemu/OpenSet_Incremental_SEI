"""汇总 Stage 34 LoRa recording-consensus seed7 完整 CIL 结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _last_r3(path: Path) -> dict[str, float]:
    """读取增量结果中 After R3 的核心指标。"""

    frame = pd.read_csv(path)
    rows = frame[frame["Stage"].astype(str).str.upper() == "AFTER R3"]
    row = rows.iloc[-1] if not rows.empty else frame.iloc[-1]
    return {
        "overall": float(row.get("Overall Acc", row.get("Overall", 0.0))),
        "old": float(row.get("Old Acc", row.get("Old", 0.0))),
        "new": float(row.get("New Acc", row.get("New", 0.0))),
        "forgetting": float(row.get("Forgetting Rate", row.get("Forgetting", 0.0))),
        "macro_f1": float(row.get("Macro F1", row.get("Macro F1 Score", 0.0))),
    }


def _clustering(path: Path) -> list[dict[str, object]]:
    """读取三轮聚类指标，确认目标簇数和无噪声约束。"""

    frame = pd.read_csv(path)
    return [
        {
            "round": str(row.get("Round", "")),
            "clusters": int(row.get("Final Cluster Count", 0)),
            "target": int(row.get("True New Classes", 0)),
            "ari": float(row.get("ARI", 0.0)),
            "hungarian": float(row.get("Hungarian Acc", 0.0)),
            "purity": float(row.get("Purity", 0.0)),
            "noise": int(row.get("Noise Points", 0)),
        }
        for _, row in frame.iterrows()
    ]


def main() -> int:
    """生成 Markdown 和 JSON，明确记录 CIL 传递结果。"""

    parser = argparse.ArgumentParser(description="汇总 Stage 34 LoRa CIL")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    r3 = _last_r3(save_dir / "incremental_results.csv")
    clustering = _clustering(save_dir / "clustering_results.csv")
    cluster_gate = all(
        item["clusters"] == item["target"] == 5 and item["noise"] == 0
        for item in clustering
    )
    baseline = {
        "overall": 0.2543,
        "old": 0.1929,
        "new": 0.5000,
    }
    summary = {
        "job_id": str(args.job_id),
        "save_dir": str(save_dir),
        "backend": "gpcc_recording_consensus",
        "consensus_threshold": 0.65,
        "r3": r3,
        "clustering": clustering,
        "cluster_gate": cluster_gate,
        "baseline_seed7": baseline,
        "delta_vs_stage27_seed7": {
            key: r3[key] - baseline[key] for key in baseline
        },
    }
    lines = [
        "# Stage 34 LoRa Recording-Consensus seed7 完整 CIL 报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 配置：LoRa Chirp + GPCC recording-consensus，阈值 `0.65` + cross-day + LoRa SSL + joint discovery-CIL。",
        "- 边界：保持 LoRa strict split；held-out IQ_8-10 不参与聚类、训练或参数选择。",
        "",
        "| 指标 | Stage 34 R3 | Stage 27 Chirp seed7 | 差值 |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("overall", "Overall"),
        ("old", "Old"),
        ("new", "New"),
    ):
        lines.append(
            f"| {label} | {r3[key]:.4f} | {baseline[key]:.4f} | "
            f"{r3[key] - baseline[key]:+.4f} |"
        )
    lines.extend(
        [
            f"| Forgetting | {r3['forgetting']:.4f} | - | - |",
            f"| Macro F1 | {r3['macro_f1']:.4f} | - | - |",
            "",
            f"- 聚类固定簇数/无噪声：`{cluster_gate}`",
            "",
            "| Round | K | Target | ARI | Hungarian | Purity | Noise |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in clustering:
        lines.append(
            f"| {item['round']} | {item['clusters']} | {item['target']} | "
            f"{item['ari']:.4f} | {item['hungarian']:.4f} | "
            f"{item['purity']:.4f} | {item['noise']} |"
        )
    lines.extend(
        [
            "",
            "当前判定：先以完整 CIL 的 Overall/Old/New 是否同步改善为准；若只改善聚类而 CIL 不改善，则转向联合 self-training，不再继续调 recording 阈值。",
            "",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Stage 34 报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
