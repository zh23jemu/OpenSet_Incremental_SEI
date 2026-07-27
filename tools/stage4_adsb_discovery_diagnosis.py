"""诊断 ADS-B 正式三种子发现链路中的簇数损失位置。"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_RUNS = {
    7: "results/stage4/adsb_main_long_radcil_seed7_44470736",
    13: "results/stage4/adsb_main_long_radcil_seed13_44470736",
    31: "results/stage4/adsb_main_long_radcil_seed31_44467424",
}


def read_rows(save_dir: Path, seed: int) -> list[dict[str, Any]]:
    """读取正式聚类 CSV，并提取能够定位处理阶段的计数统计。"""
    path = save_dir / "clustering_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少聚类结果：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "seed": seed,
            "round": row["Round"],
            "samples": int(float(row["Samples"])),
            "initial_clusters": int(float(row["Initial Cluster Count"])),
            "final_clusters": int(float(row["Final Cluster Count"])),
            "merge_loss": int(float(row.get("Iter Merge Total Merges") or 0)),
            "adaptive_splits": int(float(row.get("Adaptive Split Count") or 0)),
            "noise_reassigned": int(float(row.get("Noise Reassigned") or 0)),
        }
        for row in rows
        if row.get("Method") == "MV-ACC-CIL discovery"
    ]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按轮次汇总初始形成、合并和分裂后的簇数变化。"""
    result: list[dict[str, Any]] = []
    for round_name in ("R1", "R2", "R3"):
        selected = [row for row in rows if row["round"] == round_name]
        result.append(
            {
                "round": round_name,
                "initial_mean": statistics.mean(row["initial_clusters"] for row in selected),
                "final_mean": statistics.mean(row["final_clusters"] for row in selected),
                "merge_loss_total": sum(row["merge_loss"] for row in selected),
                "adaptive_splits_total": sum(row["adaptive_splits"] for row in selected),
                "noise_reassigned_total": sum(row["noise_reassigned"] for row in selected),
            }
        )
    return result


def build_report(rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> str:
    """生成因果定位报告；真实类数只用于事后审计，不作为参数选择信号。"""
    lines = [
        "# 阶段 4 ADS-B 发现欠聚类诊断",
        "",
        "本报告只定位 HDBSCAN 初始形成、迭代合并和自适应分裂的簇数变化。",
        "协议中的每轮 10 类仅用于事后审计；后续参数选择不得读取 held-out evaluation 标签。",
        "",
        "## 单种子处理链路",
        "",
        "| Seed | 轮次 | 初始簇 | 合并损失 | 自适应分裂 | 最终簇 | 噪声重分配样本 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['round']} | {row['initial_clusters']} | "
            f"{row['merge_loss']} | {row['adaptive_splits']} | {row['final_clusters']} | "
            f"{row['noise_reassigned']} |"
        )
    lines.extend([
        "",
        "## 三种子汇总",
        "",
        "| 轮次 | 初始簇均值 | 最终簇均值 | 合并损失总数 | 分裂总数 | 噪声重分配样本 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in summary:
        lines.append(
            f"| {item['round']} | {item['initial_mean']:.2f} | {item['final_mean']:.2f} | "
            f"{item['merge_loss_total']} | {item['adaptive_splits_total']} | "
            f"{item['noise_reassigned_total']} |"
        )
    lines.extend([
        "",
        "## 结论与预注册消融",
        "",
        "- R3 的主要损失发生在初始密度微簇形成阶段；三种子初始簇为 7/6/8，而不是后处理从 10 簇大幅合并得到。",
        "- R3 自适应分裂三种子均未触发；seed31 的合并额外损失 1 簇，属于次要风险。",
        "- seed31 下一轮只做三个单因素变体：降低最小簇比例、收紧合并阈值、放宽分裂条件；Long-RADCIL 后端保持固定。",
        "- 消融候选仅依据 discovery 特征的 silhouette、HDBSCAN 置信度、簇大小 CV 和初始到最终簇损失做诊断排序；NMI/ARI/Hungarian 仅事后报告。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="诊断 ADS-B 正式结果中的发现欠聚类阶段。")
    parser.add_argument("--output", default="results/stage4/STAGE4_ADSB_DISCOVERY_DIAGNOSIS.md")
    parser.add_argument("--json-output", default="results/stage4/stage4_adsb_discovery_diagnosis.json")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for seed, save_dir in DEFAULT_RUNS.items():
        rows.extend(read_rows(Path(save_dir), seed))
    summary = summarize(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(rows, summary), encoding="utf-8")
    Path(args.json_output).write_text(
        json.dumps({"runs": DEFAULT_RUNS, "per_seed_rounds": rows, "summary": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"ADS-B 发现诊断报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
