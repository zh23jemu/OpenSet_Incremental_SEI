"""聚合 LoRa old-logit bias 三种子结果。"""

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
    "grouped_hybrid_seed7": {"overall": 0.1495, "old": 0.0798, "new": 0.4286, "forgetting": 0.4357},
    "doi_style_seed7": {"overall": 0.1676, "old": 0.1786, "new": 0.1238, "forgetting": 0.2119},
}


def _read_r3(seed: int, root: Path) -> dict:
    """读取单个 seed 的 R3 指标和 IQ_7 bias 校准选择。"""
    run_dir = root / f"seed{seed}"
    incremental = pd.read_csv(run_dir / "incremental_results.csv")
    clustering = pd.read_csv(run_dir / "clustering_results.csv")
    calibration = pd.read_csv(run_dir / "old_logit_bias_iq7_calibration.csv")
    r3 = incremental.loc[incremental["Stage"] == "After R3"].iloc[0]
    c3 = clustering.loc[clustering["Round"] == "R3"].iloc[0]
    selected = calibration.sort_values(
        ["Stage", "IQ_7 Validation Old-Class Acc", "Old Logit Bias"],
        ascending=[True, False, True],
    ).groupby("Stage", sort=False).head(1)
    selected_r3 = selected.loc[selected["Stage"] == "After R3"].iloc[0]
    cluster_column = "Discovered Clusters" if "Discovered Clusters" in c3.index else "Final Cluster Count"
    return {
        "seed": int(seed),
        "cluster_count": int(c3[cluster_column]),
        "overall": float(r3["Overall Acc"]),
        "old": float(r3["Old Acc"]),
        "new": float(r3["New Acc"]),
        "forgetting": float(r3["Forgetting Rate"]),
        "macro_f1": float(r3["Macro F1"]),
        "r3_bias": float(selected_r3["Old Logit Bias"]),
        "r3_iq7_old": float(selected_r3["IQ_7 Validation Old-Class Acc"]),
    }


def _mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    values = pd.Series([float(row[key]) for row in rows], dtype="float64")
    return float(values.mean()), float(values.std(ddof=0))


def main() -> int:
    """生成三种子 Markdown 与 JSON 汇总。"""
    parser = argparse.ArgumentParser(description="LoRa old-logit bias 三种子报告。")
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
    adopt_candidate = bool(
        cluster_ok
        and summary["overall"]["mean"] >= REFERENCES["doi_style_seed7"]["overall"]
        and summary["old"]["mean"] > REFERENCES["grouped_hybrid_seed7"]["old"]
        and summary["new"]["mean"] > REFERENCES["doi_style_seed7"]["new"]
    )

    lines = [
        "# 阶段 5 LoRa old-logit bias 三种子报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 校准边界：每个 seed 只用 Day1 IQ_7 已知类验证集选择旧类 logit bias。",
        "- 评估边界：IQ_8-10 held-out eval 只用于最终报告，不参与 bias 选择。",
        "",
        "| Seed | R3 Clusters | Overall | Old | New | Forgetting | Macro F1 | R3 Bias | IQ_7 Old |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['cluster_count']} | {fmt(row['overall'])} | "
            f"{fmt(row['old'])} | {fmt(row['new'])} | {fmt(row['forgetting'])} | "
            f"{fmt(row['macro_f1'])} | {fmt(row['r3_bias'])} | {fmt(row['r3_iq7_old'])} |"
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
        f"- 是否采用为 LoRa 后端风险收敛候选：{'是' if adopt_candidate else '否'}。",
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
        "adopt_candidate": adopt_candidate,
    }
    output_path = Path(args.output)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"LoRa old-logit bias 三种子报告：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
