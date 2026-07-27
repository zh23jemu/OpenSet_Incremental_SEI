"""汇总 ADS-B 密度比例候选的配对三种子结果。"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"缺少配对三种子结果：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def extract(run: dict[str, Any]) -> dict[str, Any]:
    save_dir = Path(run["save_dir"])
    clusters = {row["Round"]: row for row in read_csv(save_dir / "clustering_results.csv")}
    increments = {
        row["Stage"].replace("After ", ""): row
        for row in read_csv(save_dir / "incremental_results.csv")
        if row.get("Stage", "").startswith("After R")
    }
    rounds = []
    for round_name in ("R1", "R2", "R3"):
        cluster = clusters[round_name]
        inc = increments[round_name]
        rounds.append({
            "round": round_name,
            "initial_clusters": int(number(cluster, "Initial Cluster Count")),
            "final_clusters": int(number(cluster, "Final Cluster Count")),
            "net_cluster_reduction": int(
                number(cluster, "Initial Cluster Count") - number(cluster, "Final Cluster Count")
            ),
            "label_free_silhouette": number(cluster, "Label-free Silhouette"),
            "cluster_size_cv": number(cluster, "Cluster Size CV"),
            "raw_hdbscan_confidence": number(cluster, "Raw HDBSCAN Confidence Mean"),
            "nmi_posthoc": number(cluster, "NMI"),
            "ari_posthoc": number(cluster, "ARI"),
            "hungarian_posthoc": number(cluster, "Hungarian Acc"),
            "overall_posthoc": number(inc, "Overall Acc"),
            "new_acc_posthoc": number(inc, "New Acc"),
        })
    return {"seed": run["seed"], "variant": run["variant"], "source": run["source"], "rounds": rounds}


def aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    selected = [row for row in rows if row["variant"] == variant]
    all_rounds = [item for row in selected for item in row["rounds"]]
    r3 = [next(item for item in row["rounds"] if item["round"] == "R3") for row in selected]
    return {
        "variant": variant,
        "label_free": {
            "silhouette_mean": statistics.mean(item["label_free_silhouette"] for item in all_rounds),
            "confidence_mean": statistics.mean(item["raw_hdbscan_confidence"] for item in all_rounds),
            "cluster_size_cv_mean": statistics.mean(item["cluster_size_cv"] for item in all_rounds),
            "net_cluster_reduction_total": sum(item["net_cluster_reduction"] for item in all_rounds),
        },
        "r3_posthoc": {
            "final_clusters_mean": statistics.mean(item["final_clusters"] for item in r3),
            "nmi_mean": statistics.mean(item["nmi_posthoc"] for item in r3),
            "ari_mean": statistics.mean(item["ari_posthoc"] for item in r3),
            "hungarian_mean": statistics.mean(item["hungarian_posthoc"] for item in r3),
            "overall_mean": statistics.mean(item["overall_posthoc"] for item in r3),
            "overall_std": statistics.stdev(item["overall_posthoc"] for item in r3),
            "new_acc_mean": statistics.mean(item["new_acc_posthoc"] for item in r3),
        },
    }


def build_report(rows: list[dict[str, Any]], summaries: list[dict[str, Any]], job_id: str) -> str:
    lines = [
        "# 阶段 4 ADS-B 密度比例配对三种子报告",
        "",
        f"- Slurm Job：`{job_id}`（seed7/13）；seed31 复用 Job `44474297`。",
        "- 唯一变量：MV-ACC min-cluster ratio `0.03` 对 `0.02`。",
        "- 固定项：各 seed 原长序列 checkpoint、Long-RADCIL、old:new=2.0、replay weight=3.0。",
        "",
        "## 无标签结构汇总",
        "",
        "| 配置 | Silhouette 均值 | HDBSCAN 置信度均值 | 簇大小 CV 均值 | 九轮净簇数减少 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        item = summary["label_free"]
        lines.append(
            f"| {summary['variant']} | {item['silhouette_mean']:.4f} | {item['confidence_mean']:.4f} | "
            f"{item['cluster_size_cv_mean']:.4f} | {item['net_cluster_reduction_total']} |"
        )
    lines.extend([
        "",
        "## R3 事后审计",
        "",
        "| 配置 | 最终簇均值 | NMI | ARI | Hungarian | Overall | New Acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for summary in summaries:
        item = summary["r3_posthoc"]
        lines.append(
            f"| {summary['variant']} | {item['final_clusters_mean']:.2f} | {item['nmi_mean']:.4f} | "
            f"{item['ari_mean']:.4f} | {item['hungarian_mean']:.4f} | "
            f"{item['overall_mean']:.4f}±{item['overall_std']:.4f} | {item['new_acc_mean']:.4f} |"
        )
    lines.extend([
        "",
        "## 判定边界",
        "",
        "是否锁定 ratio=0.02，先看三种子无标签 silhouette、HDBSCAN 置信度和簇大小 CV 是否整体稳定，",
        "再把真实标签聚类指标与增量准确率作为事后风险审计。若结构指标跨种子不稳定，则保留为负消融。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 ADS-B 密度比例配对三种子结果。")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    rows = [extract(run) for run in plan["runs"]]
    summaries = [aggregate(rows, variant) for variant in ("baseline_locked", "density_ratio_0p02")]
    Path(args.output).write_text(build_report(rows, summaries, args.job_id), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps({"plan": plan, "per_seed": rows, "summary": summaries}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"ADS-B 密度比例三种子报告：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
