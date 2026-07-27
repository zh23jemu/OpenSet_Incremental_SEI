"""汇总 ADS-B seed31 发现前端单因素消融结果。"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"缺少消融结果：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def extract_run(run: dict[str, Any]) -> dict[str, Any]:
    """分离无标签诊断字段与只能事后查看的真实标签指标。"""
    save_dir = Path(run["save_dir"])
    clustering = {row["Round"]: row for row in read_csv(save_dir / "clustering_results.csv")}
    incremental = {
        row["Stage"].replace("After ", ""): row
        for row in read_csv(save_dir / "incremental_results.csv")
        if row.get("Stage", "").startswith("After R")
    }
    rounds = []
    for round_name in ("R1", "R2", "R3"):
        cluster = clustering[round_name]
        inc = incremental[round_name]
        rounds.append({
            "round": round_name,
            "initial_clusters": int(number(cluster, "Initial Cluster Count")),
            "final_clusters": int(number(cluster, "Final Cluster Count")),
            "cluster_loss": int(number(cluster, "Initial Cluster Count") - number(cluster, "Final Cluster Count")),
            "label_free_silhouette": number(cluster, "Label-free Silhouette"),
            "cluster_size_cv": number(cluster, "Cluster Size CV"),
            "raw_hdbscan_confidence": number(cluster, "Raw HDBSCAN Confidence Mean"),
            "nmi_posthoc": number(cluster, "NMI"),
            "ari_posthoc": number(cluster, "ARI"),
            "hungarian_posthoc": number(cluster, "Hungarian Acc"),
            "overall_posthoc": number(inc, "Overall Acc"),
            "new_acc_posthoc": number(inc, "New Acc"),
        })
    return {
        "name": run["name"],
        "overrides": run["overrides"],
        "rounds": rounds,
        "label_free_summary": {
            "silhouette_mean": statistics.mean(row["label_free_silhouette"] for row in rounds),
            "cluster_size_cv_mean": statistics.mean(row["cluster_size_cv"] for row in rounds),
            "raw_hdbscan_confidence_mean": statistics.mean(row["raw_hdbscan_confidence"] for row in rounds),
            "cluster_loss_total": sum(row["cluster_loss"] for row in rounds),
        },
    }


def build_report(results: list[dict[str, Any]], job_id: str) -> str:
    lines = [
        "# 阶段 4 ADS-B seed31 发现单因素消融报告",
        "",
        f"- Slurm Job：`{job_id}`。",
        "- 固定项：seed31 长序列 checkpoint、Long-RADCIL、old:new=2.0、replay weight=3.0。",
        "- 下表第一部分完全不读取真实标签；第二部分仅用于实验完成后的效果审计。",
        "",
        "## 无标签发现诊断",
        "",
        "| 变体 | Silhouette 均值 | HDBSCAN 置信度均值 | 簇大小 CV 均值 | 三轮合并损失 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        item = result["label_free_summary"]
        lines.append(
            f"| {result['name']} | {item['silhouette_mean']:.4f} | "
            f"{item['raw_hdbscan_confidence_mean']:.4f} | {item['cluster_size_cv_mean']:.4f} | "
            f"{item['cluster_loss_total']} |"
        )
    lines.extend([
        "",
        "## R3 事后审计",
        "",
        "| 变体 | 初始簇 | 最终簇 | NMI | ARI | Hungarian | Overall | New Acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for result in results:
        r3 = next(row for row in result["rounds"] if row["round"] == "R3")
        lines.append(
            f"| {result['name']} | {r3['initial_clusters']} | {r3['final_clusters']} | "
            f"{r3['nmi_posthoc']:.4f} | {r3['ari_posthoc']:.4f} | {r3['hungarian_posthoc']:.4f} | "
            f"{r3['overall_posthoc']:.4f} | {r3['new_acc_posthoc']:.4f} |"
        )
    lines.extend([
        "",
        "## 判定规则",
        "",
        "本次单种子结果只用于定位敏感因素，不自动锁定正式参数。候选必须先在无标签结构指标上不出现明显退化，",
        "再扩展 seed7/13 检查跨种子稳定性；NMI、ARI、Hungarian、Overall 和 New Acc 不参与候选选择。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 ADS-B seed31 发现单因素消融。")
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
    print(f"ADS-B 发现消融报告：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
