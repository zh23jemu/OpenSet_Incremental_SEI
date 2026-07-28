"""汇总 ADS-B 保守 target split 参数消融结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


DISCOVERY_METHOD = "MV-ACC-CIL discovery"
VARIANTS = (
    "baseline_locked",
    "target_split_default",
    "target_split_max2_s026",
    "target_split_m4_s034",
    "target_split_m4_s038",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取实验结果 CSV；缺文件时直接失败，防止报告漏项。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少保守 target split 消融结果：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str, default: float = float("nan")) -> float:
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
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def extract(run: dict[str, Any]) -> dict[str, Any]:
    """提取单个 seed/variant 的三轮发现与增量指标。"""
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
            "cluster_error": int(final_clusters - target_clusters),
            "abs_cluster_error": int(abs(final_clusters - target_clusters)),
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
        "rounds": rounds,
    }


def aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    """聚合九轮无标签结构指标和 R3 held-out 审计指标。"""
    selected = [row for row in rows if row["variant"] == variant]
    all_rounds = [item for row in selected for item in row["rounds"]]
    r3 = [next(item for item in row["rounds"] if item["round"] == "R3") for row in selected]
    return {
        "variant": variant,
        "label_free": {
            "silhouette": mean_std([item["label_free_silhouette"] for item in all_rounds]),
            "confidence": mean_std([item["raw_hdbscan_confidence"] for item in all_rounds]),
            "cluster_size_cv": mean_std([item["cluster_size_cv"] for item in all_rounds]),
            "small_cluster_fraction": mean_std([item["small_cluster_fraction"] for item in all_rounds]),
            "cluster_error_total": sum(item["cluster_error"] for item in all_rounds),
            "abs_cluster_error_total": sum(item["abs_cluster_error"] for item in all_rounds),
            "target_split_total": sum(item["target_split_count"] for item in all_rounds),
        },
        "r3_posthoc": {
            "final_clusters": mean_std([item["final_clusters"] for item in r3]),
            "overall": mean_std([item["overall_posthoc"] for item in r3]),
            "old": mean_std([item["old_acc_posthoc"] for item in r3]),
            "new": mean_std([item["new_acc_posthoc"] for item in r3]),
            "forgetting": mean_std([item["forgetting_posthoc"] for item in r3]),
            "macro_f1": mean_std([item["macro_f1_posthoc"] for item in r3]),
            "nmi": mean_std([item["nmi_posthoc"] for item in r3]),
            "ari": mean_std([item["ari_posthoc"] for item in r3]),
            "hungarian": mean_std([item["hungarian_posthoc"] for item in r3]),
        },
    }


def paired_deltas(rows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    """按 seed 计算候选相对默认 target split 的 R3 差值。"""
    deltas = []
    for seed in sorted({row["seed"] for row in rows}):
        pair = {row["variant"]: row for row in rows if row["seed"] == seed}
        base = next(item for item in pair["target_split_default"]["rounds"] if item["round"] == "R3")
        cand = next(item for item in pair[variant]["rounds"] if item["round"] == "R3")
        deltas.append({
            "seed": seed,
            "variant": variant,
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
) -> tuple[str, dict[str, Any]]:
    """生成报告并输出机器可读判定摘要。"""
    lines = [
        "# 阶段 4 ADS-B 保守 target split 消融",
        "",
        f"- Slurm Job：`{job_id}`。",
        "- baseline/default target split 复用既有结果；三个保守候选为当前作业新跑。",
        "- 选择优先级：先看九轮绝对簇误差和无标签 CV/小簇比例，再看 R3 Overall/New/Forgetting 事后审计。",
        "",
        "## 无标签结构汇总",
        "",
        "| 变体 | Silhouette | HDBSCAN 置信度 | 簇大小 CV | 小簇比例 | 九轮簇误差和 | 九轮绝对簇误差 | 目标分裂总数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        item = summary["label_free"]
        lines.append(
            f"| {summary['variant']} | {metric_line(item, 'silhouette')} | "
            f"{metric_line(item, 'confidence')} | {metric_line(item, 'cluster_size_cv')} | "
            f"{metric_line(item, 'small_cluster_fraction')} | {item['cluster_error_total']} | "
            f"{item['abs_cluster_error_total']} | {item['target_split_total']} |"
        )

    lines.extend([
        "",
        "## R3 事后审计",
        "",
        "| 变体 | 最终簇 | Overall | Old | New | Forgetting | Macro F1 | NMI | ARI | Hungarian |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for summary in summaries:
        item = summary["r3_posthoc"]
        lines.append(
            f"| {summary['variant']} | {metric_line(item, 'final_clusters')} | "
            f"{metric_line(item, 'overall')} | {metric_line(item, 'old')} | "
            f"{metric_line(item, 'new')} | {metric_line(item, 'forgetting')} | "
            f"{metric_line(item, 'macro_f1')} | {metric_line(item, 'nmi')} | "
            f"{metric_line(item, 'ari')} | {metric_line(item, 'hungarian')} |"
        )

    lines.extend([
        "",
        "## 保守候选相对默认 target split 的 R3 差值",
        "",
        "| 候选 | Seed | 最终簇 | 绝对簇误差 | Silhouette | 置信度 | CV | 小簇比例 | Overall | New | Forgetting |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in deltas:
        lines.append(
            f"| {row['variant']} | {row['seed']} | {row['final_clusters_delta']:+d} | "
            f"{row['abs_cluster_error_delta']:+d} | {fmt(row['silhouette_delta'])} | "
            f"{fmt(row['confidence_delta'])} | {fmt(row['cv_delta'])} | "
            f"{fmt(row['small_fraction_delta'])} | {fmt(row['overall_delta'])} | "
            f"{fmt(row['new_delta'])} | {fmt(row['forgetting_delta'])} |"
        )

    baseline = next(item for item in summaries if item["variant"] == "baseline_locked")
    default = next(item for item in summaries if item["variant"] == "target_split_default")
    candidates = [item for item in summaries if item["variant"] not in {"baseline_locked", "target_split_default"}]
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["label_free"]["abs_cluster_error_total"],
            item["label_free"]["cluster_size_cv"]["mean"],
            -item["r3_posthoc"]["overall"]["mean"],
        ),
    )
    best = ranked[0]
    summary = {
        "baseline_abs_error": baseline["label_free"]["abs_cluster_error_total"],
        "default_abs_error": default["label_free"]["abs_cluster_error_total"],
        "default_cv_mean": default["label_free"]["cluster_size_cv"]["mean"],
        "best_candidate": best["variant"],
        "best_abs_error": best["label_free"]["abs_cluster_error_total"],
        "best_cv_mean": best["label_free"]["cluster_size_cv"]["mean"],
        "best_overall_mean": best["r3_posthoc"]["overall"]["mean"],
        "best_new_mean": best["r3_posthoc"]["new"]["mean"],
    }

    lines.extend([
        "",
        "## 自动排序提示",
        "",
        f"- baseline 九轮绝对簇误差 `{summary['baseline_abs_error']}`，默认 target split 为 `{summary['default_abs_error']}`。",
        f"- 按“绝对簇误差优先、CV 次优先、Overall 再优先”排序，当前最佳保守候选为 `{summary['best_candidate']}`。",
        "- 该排序只是无标签结构优先的辅助判断；最终仍需人工检查 seed 级 New Acc 与 Forgetting 是否退化。",
        "",
    ])
    return "\n".join(lines), summary


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 ADS-B 保守 target split 消融。")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    rows = [extract(run) for run in plan["runs"]]
    summaries = [aggregate(rows, variant) for variant in VARIANTS]
    deltas = [
        item
        for variant in VARIANTS
        if variant not in {"baseline_locked", "target_split_default"}
        for item in paired_deltas(rows, variant)
    ]
    report, decision_summary = build_report(rows, summaries, deltas, args.job_id)
    Path(args.output).write_text(report, encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "plan": plan,
                "per_seed": rows,
                "summary": summaries,
                "paired_deltas_vs_default": deltas,
                "decision_summary": decision_summary,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"ADS-B 保守 target split 消融报告：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
