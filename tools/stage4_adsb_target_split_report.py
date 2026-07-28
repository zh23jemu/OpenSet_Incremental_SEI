"""汇总 ADS-B seed31 协议目标簇数补齐分裂验证结果。

该报告器只读取实验完成后的 CSV 产物，不参与任何训练、聚类或候选选择。
无标签诊断用于判断 target split 是否缓解欠聚类且未破坏簇结构；真实标签
指标只作为事后审计，避免把评估标签反向用于参数选择。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


DISCOVERY_METHOD = "MV-ACC-CIL discovery"


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取单个实验目录下的结果 CSV，并在缺失时给出明确错误。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少结果文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    """将 CSV 单元格安全转为浮点数，兼容历史结果中不存在的新诊断字段。"""
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def fmt(value: float) -> str:
    """Markdown 表格数值格式化；缺失值保持为 NA，避免误读为 0。"""
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.4f}"


def extract_run(run: dict[str, Any]) -> dict[str, Any]:
    """提取每个候选三轮发现诊断和增量识别结果。"""
    save_dir = Path(run["save_dir"])
    clustering_rows = [
        row for row in read_csv(save_dir / "clustering_results.csv")
        if row.get("Method") == DISCOVERY_METHOD
    ]
    if not clustering_rows:
        raise ValueError(f"{save_dir} 中找不到 {DISCOVERY_METHOD} 行")
    clustering = {row["Round"]: row for row in clustering_rows}
    incremental = {
        row["Stage"].replace("After ", ""): row
        for row in read_csv(save_dir / "incremental_results.csv")
        if row.get("Stage", "").startswith("After R")
    }

    rounds = []
    for round_name in ("R1", "R2", "R3"):
        cluster = clustering[round_name]
        inc = incremental[round_name]
        final_clusters = int(number(cluster, "Final Cluster Count"))
        target_clusters = int(number(cluster, "True New Classes"))
        rounds.append({
            "round": round_name,
            "initial_clusters": int(number(cluster, "Initial Cluster Count")),
            "final_clusters": final_clusters,
            "target_clusters": target_clusters,
            "cluster_deficit": int(target_clusters - final_clusters),
            "target_split_enabled": bool(str(cluster.get("Target Split Enabled", "False")).lower() == "true"),
            "target_split_count": int(number(cluster, "Target Split Count", 0.0)),
            "target_split_silhouette_mean": number(cluster, "Target Split Silhouette Mean"),
            "label_free_silhouette": number(cluster, "Label-free Silhouette"),
            "cluster_size_cv": number(cluster, "Cluster Size CV"),
            "small_cluster_fraction": number(cluster, "Small Cluster Fraction"),
            "raw_hdbscan_confidence": number(cluster, "Raw HDBSCAN Confidence Mean"),
            "nmi_posthoc": number(cluster, "NMI"),
            "ari_posthoc": number(cluster, "ARI"),
            "hungarian_posthoc": number(cluster, "Hungarian Acc"),
            "overall_posthoc": number(inc, "Overall Acc"),
            "old_acc_posthoc": number(inc, "Old Acc"),
            "new_acc_posthoc": number(inc, "New Acc"),
            "forgetting_posthoc": number(inc, "Forgetting Rate"),
        })

    return {
        "name": run["name"],
        "save_dir": run["save_dir"],
        "overrides": run.get("overrides", {}),
        "rounds": rounds,
        "label_free_summary": {
            "silhouette_mean": statistics.mean(row["label_free_silhouette"] for row in rounds),
            "cluster_size_cv_mean": statistics.mean(row["cluster_size_cv"] for row in rounds),
            "small_cluster_fraction_mean": statistics.mean(row["small_cluster_fraction"] for row in rounds),
            "raw_hdbscan_confidence_mean": statistics.mean(row["raw_hdbscan_confidence"] for row in rounds),
            "cluster_deficit_total": sum(row["cluster_deficit"] for row in rounds),
            "target_split_total": sum(row["target_split_count"] for row in rounds),
        },
    }


def build_report(results: list[dict[str, Any]], job_id: str) -> str:
    """生成面向阶段 4 风险收敛的简洁 Markdown 报告。"""
    lines = [
        "# 阶段 4 ADS-B seed31 目标簇数补齐分裂验证",
        "",
        f"- Slurm Job：`{job_id}`。",
        "- 固定项：seed31、ADS-B long checkpoint、Long-RADCIL、old:new=2.0、replay weight=3.0。",
        "- 变量：是否启用协议目标簇数补齐分裂；目标簇数来自公开协议 round_size=10，不读取真实标签。",
        "",
        "## 无标签结构诊断",
        "",
        "| 变体 | Silhouette 均值 | HDBSCAN 置信度均值 | 簇大小 CV 均值 | 小簇比例均值 | 三轮缺簇总数 | 目标分裂总数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        item = result["label_free_summary"]
        lines.append(
            f"| {result['name']} | {fmt(item['silhouette_mean'])} | "
            f"{fmt(item['raw_hdbscan_confidence_mean'])} | {fmt(item['cluster_size_cv_mean'])} | "
            f"{fmt(item['small_cluster_fraction_mean'])} | {item['cluster_deficit_total']} | "
            f"{item['target_split_total']} |"
        )

    lines.extend([
        "",
        "## 逐轮发现审计",
        "",
        "| 变体 | 轮次 | 初始簇 | 最终簇 | 缺簇数 | 目标分裂数 | 分裂 Silhouette | Label-free Silhouette |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for result in results:
        for row in result["rounds"]:
            lines.append(
                f"| {result['name']} | {row['round']} | {row['initial_clusters']} | "
                f"{row['final_clusters']} | {row['cluster_deficit']} | {row['target_split_count']} | "
                f"{fmt(row['target_split_silhouette_mean'])} | {fmt(row['label_free_silhouette'])} |"
            )

    lines.extend([
        "",
        "## R3 事后审计",
        "",
        "| 变体 | 最终簇 | NMI | ARI | Hungarian | Overall | Old | New | Forgetting |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for result in results:
        r3 = next(row for row in result["rounds"] if row["round"] == "R3")
        lines.append(
            f"| {result['name']} | {r3['final_clusters']} | {fmt(r3['nmi_posthoc'])} | "
            f"{fmt(r3['ari_posthoc'])} | {fmt(r3['hungarian_posthoc'])} | "
            f"{fmt(r3['overall_posthoc'])} | {fmt(r3['old_acc_posthoc'])} | "
            f"{fmt(r3['new_acc_posthoc'])} | {fmt(r3['forgetting_posthoc'])} |"
        )

    lines.extend([
        "",
        "## 判定口径",
        "",
        "若 target_split 能显著降低缺簇数，同时无标签 silhouette、HDBSCAN 置信度、簇大小 CV 和小簇比例不明显退化，",
        "才考虑扩展 seed7/13；真实标签指标只用于事后确认风险方向，不作为本轮候选选择依据。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 ADS-B 目标簇数补齐分裂验证。")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    results = [extract_run(run) for run in plan["runs"]]
    Path(args.output).write_text(build_report(results, args.job_id), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps({"plan": plan, "results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"ADS-B 目标分裂验证报告：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
