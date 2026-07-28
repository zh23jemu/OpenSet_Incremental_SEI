"""汇总 ADS-B 目标簇数补齐分裂的配对三种子验证结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


DISCOVERY_METHOD = "MV-ACC-CIL discovery"
VARIANTS = ("baseline_locked", "target_split")


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取结果 CSV，缺失时直接失败，避免报告静默漏 seed。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少 target split 三种子结果：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    """兼容历史 CSV 中不存在的新字段。"""
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def fmt(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.4f}"


def mean_std(values: list[float]) -> dict[str, float]:
    """返回均值和样本标准差；三种子场景使用样本标准差。"""
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def extract(run: dict[str, Any]) -> dict[str, Any]:
    """从单个运行目录提取三轮发现与增量识别指标。"""
    save_dir = Path(run["save_dir"])
    clustering_rows = [
        row for row in read_csv(save_dir / "clustering_results.csv")
        if row.get("Method") == DISCOVERY_METHOD
    ]
    if not clustering_rows:
        raise ValueError(f"{save_dir} 中找不到 {DISCOVERY_METHOD} 行")
    clusters = {row["Round"]: row for row in clustering_rows}
    increments = {
        row["Stage"].replace("After ", ""): row
        for row in read_csv(save_dir / "incremental_results.csv")
        if row.get("Stage", "").startswith("After R")
    }

    rounds = []
    for round_name in ("R1", "R2", "R3"):
        cluster = clusters[round_name]
        inc = increments[round_name]
        target_clusters = int(number(cluster, "True New Classes"))
        final_clusters = int(number(cluster, "Final Cluster Count"))
        rounds.append({
            "round": round_name,
            "initial_clusters": int(number(cluster, "Initial Cluster Count")),
            "final_clusters": final_clusters,
            "target_clusters": target_clusters,
            "cluster_deficit": int(target_clusters - final_clusters),
            "abs_cluster_error": int(abs(target_clusters - final_clusters)),
            "target_split_count": int(number(cluster, "Target Split Count", 0.0)),
            "target_split_silhouette_mean": number(cluster, "Target Split Silhouette Mean"),
            "label_free_silhouette": number(cluster, "Label-free Silhouette"),
            "raw_hdbscan_confidence": number(cluster, "Raw HDBSCAN Confidence Mean"),
            "cluster_size_cv": number(cluster, "Cluster Size CV"),
            "small_cluster_fraction": number(cluster, "Small Cluster Fraction"),
            "nmi_posthoc": number(cluster, "NMI"),
            "ari_posthoc": number(cluster, "ARI"),
            "hungarian_posthoc": number(cluster, "Hungarian Acc"),
            "overall_posthoc": number(inc, "Overall Acc"),
            "old_acc_posthoc": number(inc, "Old Acc"),
            "new_acc_posthoc": number(inc, "New Acc"),
            "forgetting_posthoc": number(inc, "Forgetting Rate"),
            "macro_f1_posthoc": number(inc, "Macro F1"),
        })

    return {
        "seed": int(run["seed"]),
        "variant": run["variant"],
        "source": run["source"],
        "save_dir": run["save_dir"],
        "rounds": rounds,
    }


def aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    """聚合某个变体的九轮无标签指标和 R3 事后指标。"""
    selected = [row for row in rows if row["variant"] == variant]
    all_rounds = [item for row in selected for item in row["rounds"]]
    r3_rows = [next(item for item in row["rounds"] if item["round"] == "R3") for row in selected]
    return {
        "variant": variant,
        "label_free": {
            "silhouette": mean_std([item["label_free_silhouette"] for item in all_rounds]),
            "confidence": mean_std([item["raw_hdbscan_confidence"] for item in all_rounds]),
            "cluster_size_cv": mean_std([item["cluster_size_cv"] for item in all_rounds]),
            "small_cluster_fraction": mean_std([item["small_cluster_fraction"] for item in all_rounds]),
            "cluster_deficit_total": sum(item["cluster_deficit"] for item in all_rounds),
            "abs_cluster_error_total": sum(item["abs_cluster_error"] for item in all_rounds),
            "target_split_total": sum(item["target_split_count"] for item in all_rounds),
        },
        "r3_posthoc": {
            "final_clusters": mean_std([item["final_clusters"] for item in r3_rows]),
            "nmi": mean_std([item["nmi_posthoc"] for item in r3_rows]),
            "ari": mean_std([item["ari_posthoc"] for item in r3_rows]),
            "hungarian": mean_std([item["hungarian_posthoc"] for item in r3_rows]),
            "overall": mean_std([item["overall_posthoc"] for item in r3_rows]),
            "old": mean_std([item["old_acc_posthoc"] for item in r3_rows]),
            "new": mean_std([item["new_acc_posthoc"] for item in r3_rows]),
            "forgetting": mean_std([item["forgetting_posthoc"] for item in r3_rows]),
            "macro_f1": mean_std([item["macro_f1_posthoc"] for item in r3_rows]),
        },
    }


def paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 seed 计算 target_split 相对 baseline 的 R3 差值。"""
    deltas = []
    for seed in sorted({row["seed"] for row in rows}):
        pair = {row["variant"]: row for row in rows if row["seed"] == seed}
        base = next(item for item in pair["baseline_locked"]["rounds"] if item["round"] == "R3")
        cand = next(item for item in pair["target_split"]["rounds"] if item["round"] == "R3")
        deltas.append({
            "seed": seed,
            "final_clusters_delta": cand["final_clusters"] - base["final_clusters"],
            "abs_cluster_error_delta": cand["abs_cluster_error"] - base["abs_cluster_error"],
            "silhouette_delta": cand["label_free_silhouette"] - base["label_free_silhouette"],
            "confidence_delta": cand["raw_hdbscan_confidence"] - base["raw_hdbscan_confidence"],
            "cv_delta": cand["cluster_size_cv"] - base["cluster_size_cv"],
            "small_fraction_delta": cand["small_cluster_fraction"] - base["small_cluster_fraction"],
            "overall_delta": cand["overall_posthoc"] - base["overall_posthoc"],
            "old_delta": cand["old_acc_posthoc"] - base["old_acc_posthoc"],
            "new_delta": cand["new_acc_posthoc"] - base["new_acc_posthoc"],
            "forgetting_delta": cand["forgetting_posthoc"] - base["forgetting_posthoc"],
        })
    return deltas


def metric_line(summary: dict[str, Any], key: str) -> str:
    item = summary[key]
    return f"{fmt(item['mean'])}±{fmt(item['std'])}"


def build_report(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    job_id: str,
) -> str:
    """生成 ADS-B target split 三种子风险收敛报告。"""
    lines = [
        "# 阶段 4 ADS-B target split 配对三种子验证",
        "",
        f"- Slurm Job：`{job_id}`（seed7/13）；seed31 复用 Job `44670427`。",
        "- 唯一变量：是否启用协议目标簇数补齐分裂。",
        "- 固定项：各 seed 长序列 checkpoint、Long-RADCIL、old:new=2.0、replay weight=3.0、MV-ACC ratio=0.03。",
        "",
        "## 无标签结构汇总",
        "",
        "| 变体 | Silhouette | HDBSCAN 置信度 | 簇大小 CV | 小簇比例 | 九轮缺簇总数 | 九轮绝对簇误差 | 目标分裂总数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        item = summary["label_free"]
        lines.append(
            f"| {summary['variant']} | {metric_line(item, 'silhouette')} | "
            f"{metric_line(item, 'confidence')} | {metric_line(item, 'cluster_size_cv')} | "
            f"{metric_line(item, 'small_cluster_fraction')} | {item['cluster_deficit_total']} | "
            f"{item['abs_cluster_error_total']} | {item['target_split_total']} |"
        )

    lines.extend([
        "",
        "## R3 事后审计",
        "",
        "| 变体 | 最终簇 | NMI | ARI | Hungarian | Overall | Old | New | Forgetting | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for summary in summaries:
        item = summary["r3_posthoc"]
        lines.append(
            f"| {summary['variant']} | {metric_line(item, 'final_clusters')} | "
            f"{metric_line(item, 'nmi')} | {metric_line(item, 'ari')} | "
            f"{metric_line(item, 'hungarian')} | {metric_line(item, 'overall')} | "
            f"{metric_line(item, 'old')} | {metric_line(item, 'new')} | "
            f"{metric_line(item, 'forgetting')} | {metric_line(item, 'macro_f1')} |"
        )

    lines.extend([
        "",
        "## R3 配对差值",
        "",
        "| Seed | 最终簇差值 | 绝对簇误差差值 | Silhouette | 置信度 | CV | 小簇比例 | Overall | Old | New | Forgetting |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in deltas:
        lines.append(
            f"| {row['seed']} | {row['final_clusters_delta']:+d} | {row['abs_cluster_error_delta']:+d} | "
            f"{fmt(row['silhouette_delta'])} | {fmt(row['confidence_delta'])} | "
            f"{fmt(row['cv_delta'])} | {fmt(row['small_fraction_delta'])} | "
            f"{fmt(row['overall_delta'])} | {fmt(row['old_delta'])} | "
            f"{fmt(row['new_delta'])} | {fmt(row['forgetting_delta'])} |"
        )

    baseline = next(item for item in summaries if item["variant"] == "baseline_locked")
    candidate = next(item for item in summaries if item["variant"] == "target_split")
    overall_delta = (
        candidate["r3_posthoc"]["overall"]["mean"] - baseline["r3_posthoc"]["overall"]["mean"]
    )
    new_delta = candidate["r3_posthoc"]["new"]["mean"] - baseline["r3_posthoc"]["new"]["mean"]
    error_delta = (
        candidate["label_free"]["abs_cluster_error_total"]
        - baseline["label_free"]["abs_cluster_error_total"]
    )
    lines.extend([
        "",
        "## 判定口径",
        "",
        f"- 九轮绝对簇误差变化 `{error_delta:+d}`。",
        f"- R3 Overall 均值变化 `{overall_delta:+.4f}`，New Acc 均值变化 `{new_delta:+.4f}`。",
        "- 若三种子同时降低缺簇/簇误差，且无标签结构指标没有系统性退化，则 target split 可作为 ADS-B 欠聚类风险收敛候选。",
        "- 若 seed7/13 出现过切分、簇大小失衡或旧类/遗忘明显退化，则只保留为 seed31 正消融，不升级为默认配置。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 ADS-B target split 三种子配对验证。")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    rows = [extract(run) for run in plan["runs"]]
    summaries = [aggregate(rows, variant) for variant in VARIANTS]
    deltas = paired_deltas(rows)
    Path(args.output).write_text(build_report(rows, summaries, deltas, args.job_id), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {"plan": plan, "per_seed": rows, "summary": summaries, "paired_deltas": deltas},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"ADS-B target split 三种子报告：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
