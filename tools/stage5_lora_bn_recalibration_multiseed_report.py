"""聚合 LoRa BatchNorm 重校准三种子结果。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.stage2_wisig_main_report import fmt


REFERENCES = {
    "selected_radcil_seed7": {"overall": 0.1419, "old": 0.0798, "new": 0.3905, "forgetting": 0.5190},
    "grouped_hybrid_seed7": {"overall": 0.1495, "old": 0.0798, "new": 0.4286, "forgetting": 0.4357},
    "old_logit_bias_multiseed": {"overall": 0.1714, "old": 0.2016, "new": 0.0508, "forgetting": 0.3175},
}


def _read_r3(seed: int, root: Path) -> dict:
    """读取单个 seed 的 R3 指标。

    BN 重校准没有额外的 IQ_7 选参表，因此这里只读取 discovery 簇数和
    最终增量分类指标。列名兼容旧版与 target-split 版本的聚类结果表。
    """
    run_dir = root / f"seed{seed}"
    incremental = pd.read_csv(run_dir / "incremental_results.csv")
    clustering = pd.read_csv(run_dir / "clustering_results.csv")
    r3 = incremental.loc[incremental["Stage"] == "After R3"].iloc[0]
    c3 = clustering.loc[clustering["Round"] == "R3"].iloc[0]
    cluster_column = "Discovered Clusters" if "Discovered Clusters" in c3.index else "Final Cluster Count"
    return {
        "seed": int(seed),
        "cluster_count": int(c3[cluster_column]),
        "overall": float(r3["Overall Acc"]),
        "old": float(r3["Old Acc"]),
        "new": float(r3["New Acc"]),
        "forgetting": float(r3["Forgetting Rate"]),
        "macro_f1": float(r3["Macro F1"]),
    }


def _mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    """返回总体均值和总体标准差，用于和项目内其它三种子报告口径一致。"""
    values = pd.Series([float(row[key]) for row in rows], dtype="float64")
    return float(values.mean()), float(values.std(ddof=0))


def main() -> int:
    """生成三种子 Markdown 与 JSON 汇总。"""
    parser = argparse.ArgumentParser(description="LoRa BN 重校准三种子报告。")
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--seeds", default="7,13,31")
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.matrix_root)
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    rows = [_read_r3(seed, root) for seed in seeds]
    summary = {
        key: {"mean": _mean_std(rows, key)[0], "std": _mean_std(rows, key)[1]}
        for key in ["overall", "old", "new", "forgetting", "macro_f1"]
    }
    cluster_ok = all(int(row["cluster_count"]) == 5 for row in rows)
    balanced_candidate = bool(
        cluster_ok
        and summary["overall"]["mean"] >= REFERENCES["grouped_hybrid_seed7"]["overall"]
        and summary["old"]["mean"] > REFERENCES["selected_radcil_seed7"]["old"]
        and summary["new"]["mean"] >= REFERENCES["selected_radcil_seed7"]["new"] - 0.02
        and summary["new"]["mean"] > REFERENCES["old_logit_bias_multiseed"]["new"]
    )

    lines = [
        "# 阶段 5 LoRa BatchNorm 重校准三种子报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 边界：BN 只用 replay memory 和当前 discovery/enrollment 样本刷新统计。",
        "- 评估边界：IQ_8-10 held-out eval 只用于最终报告，不参与 BN 统计估计或选参。",
        "",
        "| Seed | R3 Clusters | Overall | Old | New | Forgetting | Macro F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['cluster_count']} | {fmt(row['overall'])} | "
            f"{fmt(row['old'])} | {fmt(row['new'])} | {fmt(row['forgetting'])} | "
            f"{fmt(row['macro_f1'])} |"
        )
    lines.extend([
        "",
        "## 三种子均值",
        "",
        "| 指标 | Mean | Std |",
        "| --- | ---: | ---: |",
    ])
    for key, label in [
        ("overall", "Overall"),
        ("old", "Old"),
        ("new", "New"),
        ("forgetting", "Forgetting"),
        ("macro_f1", "Macro F1"),
    ]:
        lines.append(f"| {label} | {fmt(summary[key]['mean'])} | {fmt(summary[key]['std'])} |")
    lines.extend([
        "",
        "## 判定",
        "",
        f"- 三种子 R3 均为 5 簇：{'通过' if cluster_ok else '未通过'}。",
        f"- 是否作为 LoRa 平衡后端候选：{'是' if balanced_candidate else '否'}。",
        "",
    ])

    payload = {
        "job_id": args.job_id,
        "matrix_root": str(root),
        "seeds": seeds,
        "rows": rows,
        "summary": summary,
        "references": REFERENCES,
        "cluster_count_gate_passed": cluster_ok,
        "balanced_candidate": balanced_candidate,
    }
    output_path = Path(args.output)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"LoRa BN 重校准三种子报告：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
