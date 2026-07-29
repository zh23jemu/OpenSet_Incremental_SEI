"""生成 LoRa old-logit bias seed7 诊断报告。"""

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
    "selected_radcil": {"overall": 0.1419, "old": 0.0798, "new": 0.3905, "forgetting": 0.5190},
    "grouped_hybrid": {"overall": 0.1495, "old": 0.0798, "new": 0.4286, "forgetting": 0.4357},
    "doi_style": {"overall": 0.1676, "old": 0.1786, "new": 0.1238, "forgetting": 0.2119},
}


def main() -> int:
    """汇总 old-logit bias 结果，并判断它是否值得扩展三种子。

    该报告只读取本次作业产出的 CSV。old-logit bias 的选择依据来自
    Day1/IQ_7 验证集，held-out IQ_8-10 只在训练完成后用于一次性评估。
    """
    parser = argparse.ArgumentParser(description="LoRa old-logit bias seed7 报告。")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    metrics = extract_round_metrics(save_dir)
    r3 = next(row for row in metrics["rounds"] if row["round"] == "R3")
    calibration = pd.read_csv(save_dir / "old_logit_bias_iq7_calibration.csv")
    selected = calibration.sort_values(
        ["Stage", "IQ_7 Validation Old-Class Acc", "Old Logit Bias"],
        ascending=[True, False, True],
    ).groupby("Stage", sort=False).head(1)
    cluster_ok = all(int(row["cluster_count"]) == 5 for row in metrics["rounds"])
    old_improved = float(r3["old_acc"]) > REFERENCES["grouped_hybrid"]["old"]
    overall_beats_doi = float(r3["overall_acc"]) > REFERENCES["doi_style"]["overall"]
    new_not_collapsed = float(r3["new_acc"]) > REFERENCES["doi_style"]["new"]
    expand_multiseed = bool(cluster_ok and old_improved and overall_beats_doi and new_not_collapsed)

    lines = [
        "# 阶段 5 LoRa old-logit bias seed7 诊断报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 目标：验证 LoRa 低分是否主要来自旧类 logits 被新增类头压低。",
        "- 校准边界：bias 只用 Day1 IQ_7 已知类验证集选择；IQ_8-10 held-out eval 不参与选参。",
        "- 发现边界：沿用协议预声明的每轮 5 类目标簇约束，不读取未知轮次真值做聚类选择。",
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

    selected_text = "；".join(
        f"{row['Stage']}=bias {fmt(row['Old Logit Bias'])}, IQ_7 old {fmt(row['IQ_7 Validation Old-Class Acc'])}"
        for _, row in selected.iterrows()
    )
    lines.extend([
        "",
        "## 判定",
        "",
        f"- 三轮目标簇数：{'通过' if cluster_ok else '未通过'}。",
        f"- R3 Old 是否超过 grouped hybrid：{'通过' if old_improved else '未通过'}。",
        f"- R3 Overall 是否超过 DOI-style：{'通过' if overall_beats_doi else '未通过'}。",
        f"- R3 New 是否未塌缩到 DOI-style 以下：{'通过' if new_not_collapsed else '未通过'}。",
        f"- 是否扩展 seed13/31：{'是' if expand_multiseed else '否'}。",
        f"- IQ_7 各阶段最优 bias：{selected_text}。",
        "",
    ])

    summary = {
        "job_id": args.job_id,
        "save_dir": str(save_dir),
        "metrics": metrics,
        "references": REFERENCES,
        "cluster_count_gate_passed": cluster_ok,
        "old_improved_over_grouped_hybrid": old_improved,
        "overall_beats_doi_style": overall_beats_doi,
        "new_not_collapsed_below_doi_style": new_not_collapsed,
        "expand_multiseed": expand_multiseed,
    }
    output_path = Path(args.output)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"LoRa old-logit bias 报告：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
