"""生成 LoRa 目标簇数约束与分组双头融合的 seed7 对比报告。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.stage2_wisig_main_report import extract_round_metrics, fmt


REFERENCES = {
    "RADCIL": {"overall": 0.1419, "old": 0.0798, "new": 0.3905, "forgetting": 0.5190},
    "DOI-style": {"overall": 0.1676, "old": 0.1786, "new": 0.1238, "forgetting": 0.2119},
}


def main() -> int:
    """汇总三轮结果，并按预先声明的旧类、新类与聚类门槛给出结论。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    metrics = extract_round_metrics(save_dir)
    r3 = next(row for row in metrics["rounds"] if row["round"] == "R3")
    calibration = pd.read_csv(save_dir / "grouped_doi_iq7_calibration.csv")
    cluster_ok = all(int(row["cluster_count"]) == 5 for row in metrics["rounds"])
    tradeoff_ok = (
        float(r3["old_acc"]) > REFERENCES["RADCIL"]["old"]
        and float(r3["new_acc"]) > REFERENCES["DOI-style"]["new"]
        and float(r3["forgetting_rate"]) < REFERENCES["RADCIL"]["forgetting"]
    )
    expand_multiseed = bool(cluster_ok and tradeoff_ok)

    lines = [
        "# 阶段 5 LoRa 分组双头融合 seed7 报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 校准边界：融合权重只使用 Day1 IQ_7 已知类验证集；IQ_8-10 仅用于最终评估。",
        "- 发现约束：每轮仅使用协议预先声明的 5 个新增类目标，将过聚类的最近多视图原型合并至 5 簇。",
        "",
        "| 轮次 | Cluster Count | Overall | Old | New | Forgetting | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics["rounds"]:
        lines.append(
            f"| {row['round']} | {int(row['cluster_count'])} | {fmt(row['overall_acc'])} | "
            f"{fmt(row['old_acc'])} | {fmt(row['new_acc'])} | "
            f"{fmt(row['forgetting_rate'])} | {fmt(row['macro_f1'])} |"
        )
    selected = calibration.sort_values(
        ["Stage", "Validation Accuracy", "Fusion Weight"],
        ascending=[True, False, True],
    ).groupby("Stage", sort=False).head(1)
    selected_text = "；".join(
        f"{row['Stage']}=weight {fmt(row['Fusion Weight'])}, acc {fmt(row['Validation Accuracy'])}"
        for _, row in selected.iterrows()
    )
    lines.extend([
        "",
        "## 判定",
        "",
        f"- 三轮目标簇数约束：{'通过' if cluster_ok else '未通过'}。",
        f"- 新旧类权衡门槛：{'通过' if tradeoff_ok else '未通过'}。",
        f"- 是否扩展 seed13/31：{'是' if expand_multiseed else '否'}。",
        f"- IQ_7 各阶段最优候选：{selected_text}。",
        "",
    ])

    summary = {
        "job_id": args.job_id,
        "save_dir": str(save_dir),
        "metrics": metrics,
        "references": REFERENCES,
        "cluster_count_gate_passed": cluster_ok,
        "old_new_tradeoff_gate_passed": tradeoff_ok,
        "expand_multiseed": expand_multiseed,
    }
    output_path = Path(args.output)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"LoRa 分组双头融合报告：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
