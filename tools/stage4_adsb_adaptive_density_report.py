"""汇总 ADS-B 无标签轮次自适应密度三种子结果。"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"缺少结果文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def after_round(rows: list[dict[str, str]], round_name: str) -> dict[str, str]:
    target = f"After {round_name}"
    return next(row for row in rows if row.get("Stage") == target)


def cluster_round(rows: list[dict[str, str]], round_name: str) -> dict[str, str]:
    return next(row for row in rows if row.get("Round") == round_name)


def extract(run: dict[str, Any]) -> dict[str, Any]:
    save_dir = Path(run["save_dir"])
    baseline_dir = Path(run["baseline_dir"])
    audit = read_rows(save_dir / "mvacc_adaptive_density_audit.csv")
    adaptive_inc = read_rows(save_dir / "incremental_results.csv")
    adaptive_cluster = read_rows(save_dir / "clustering_results.csv")
    baseline_inc = read_rows(baseline_dir / "incremental_results.csv")
    baseline_cluster = read_rows(baseline_dir / "clustering_results.csv")

    rounds = []
    for round_name in ("R1", "R2", "R3"):
        anchor_audit = next(
            row for row in audit
            if row["Round"] == round_name and row["Source"] == "anchor"
        )
        adaptive_row = cluster_round(adaptive_cluster, round_name)
        baseline_row = cluster_round(baseline_cluster, round_name)
        rounds.append({
            "round": round_name,
            "selected_ratio": float(anchor_audit["Selected Ratio"]),
            "selected_source": anchor_audit["Selected Source"],
            "rejected_by": anchor_audit["Rejected By"],
            "partition_agreement_ari": float(anchor_audit["Partition Agreement ARI"]),
            "adaptive_clusters": int(float(adaptive_row["Final Cluster Count"])),
            "baseline_clusters": int(float(baseline_row["Final Cluster Count"])),
            "adaptive_silhouette": float(adaptive_row["Label-free Silhouette"]),
            "baseline_silhouette": float(baseline_row["Label-free Silhouette"]),
        })

    adaptive_r3 = after_round(adaptive_inc, "R3")
    baseline_r3 = after_round(baseline_inc, "R3")
    return {
        "seed": int(run["seed"]),
        "rounds": rounds,
        "adaptive_r3": {
            "overall": float(adaptive_r3["Overall Acc"]),
            "old": float(adaptive_r3["Old Acc"]),
            "new": float(adaptive_r3["New Acc"]),
            "forgetting": float(adaptive_r3["Forgetting Rate"]),
            "macro_f1": float(adaptive_r3["Macro F1"]),
        },
        "baseline_r3": {
            "overall": float(baseline_r3["Overall Acc"]),
            "old": float(baseline_r3["Old Acc"]),
            "new": float(baseline_r3["New Acc"]),
            "forgetting": float(baseline_r3["Forgetting Rate"]),
            "macro_f1": float(baseline_r3["Macro F1"]),
        },
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.pstdev(values)


def aggregate(results: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    summary = {}
    for metric in ("overall", "old", "new", "forgetting", "macro_f1"):
        mean, std = mean_std([row[key][metric] for row in results])
        summary[metric] = {"mean": mean, "std": std}
    return summary


def build_report(results: list[dict[str, Any]], job_id: str) -> tuple[str, dict[str, Any]]:
    adaptive = aggregate(results, "adaptive_r3")
    baseline = aggregate(results, "baseline_r3")
    selected_candidate = sum(
        row["selected_source"] == "candidate"
        for result in results
        for row in result["rounds"]
    )
    summary = {
        "job_id": job_id,
        "candidate_selected_rounds": selected_candidate,
        "total_rounds": len(results) * 3,
        "adaptive_r3": adaptive,
        "baseline_r3": baseline,
        "r3_delta": {
            metric: adaptive[metric]["mean"] - baseline[metric]["mean"]
            for metric in adaptive
        },
    }
    lines = [
        "# 阶段 4 ADS-B 无标签自适应密度三种子报告",
        "",
        f"- Slurm Job：`{job_id}`。",
        "- 锚点 ratio 0.03；候选 ratio 0.02；Long-RADCIL 后端保持不变。",
        "- 选择门控不读取真实设备标签、真实新类数量或 held-out evaluation 指标。",
        "",
        "## 每轮选择审计",
        "",
        "| Seed | 轮次 | 选择 ratio | 来源 | 最终簇 | 锚点簇 | Silhouette | 锚点 Silhouette | 候选间 ARI | 拒绝原因 |",
        "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        for row in result["rounds"]:
            lines.append(
                f"| {result['seed']} | {row['round']} | {row['selected_ratio']:.2f} | "
                f"{row['selected_source']} | {row['adaptive_clusters']} | {row['baseline_clusters']} | "
                f"{row['adaptive_silhouette']:.4f} | {row['baseline_silhouette']:.4f} | "
                f"{row['partition_agreement_ari']:.4f} | {row['rejected_by'] or '-'} |"
            )
    lines.extend([
        "",
        "## R3 正式比较",
        "",
        "| 配置 | Overall | Old | New | Forgetting | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| 固定 0.03 | {baseline['overall']['mean']:.4f}±{baseline['overall']['std']:.4f} | "
            f"{baseline['old']['mean']:.4f}±{baseline['old']['std']:.4f} | "
            f"{baseline['new']['mean']:.4f}±{baseline['new']['std']:.4f} | "
            f"{baseline['forgetting']['mean']:.4f}±{baseline['forgetting']['std']:.4f} | "
            f"{baseline['macro_f1']['mean']:.4f}±{baseline['macro_f1']['std']:.4f} |"
        ),
        (
            f"| 自适应 | {adaptive['overall']['mean']:.4f}±{adaptive['overall']['std']:.4f} | "
            f"{adaptive['old']['mean']:.4f}±{adaptive['old']['std']:.4f} | "
            f"{adaptive['new']['mean']:.4f}±{adaptive['new']['std']:.4f} | "
            f"{adaptive['forgetting']['mean']:.4f}±{adaptive['forgetting']['std']:.4f} | "
            f"{adaptive['macro_f1']['mean']:.4f}±{adaptive['macro_f1']['std']:.4f} |"
        ),
        "",
        "## 判定",
        "",
        f"候选在 `{selected_candidate}/{len(results) * 3}` 个轮次通过全部无标签门控。",
        f"R3 Overall 相对固定 0.03 为 `{summary['r3_delta']['overall']:+.4f}`，New Acc 为 `{summary['r3_delta']['new']:+.4f}`，",
        f"Forgetting 为 `{summary['r3_delta']['forgetting']:+.4f}`。若收益不稳定，本策略按负消融归档并停止继续调密度参数。",
        "",
    ])
    return "\n".join(lines), summary


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 ADS-B 自适应密度三种子结果。")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    results = [extract(run) for run in plan["runs"]]
    report, summary = build_report(results, args.job_id)
    Path(args.output).write_text(report, encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps({"plan": plan, "per_seed": results, "summary": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"ADS-B 自适应密度报告：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
