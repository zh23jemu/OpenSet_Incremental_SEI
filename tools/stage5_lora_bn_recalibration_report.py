"""生成 LoRa BatchNorm 重校准 seed7 诊断报告。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.stage2_wisig_main_report import extract_round_metrics, fmt


REFERENCES = {
    "selected_radcil": {"overall": 0.1419, "old": 0.0798, "new": 0.3905, "forgetting": 0.5190},
    "grouped_hybrid": {"overall": 0.1495, "old": 0.0798, "new": 0.4286, "forgetting": 0.4357},
    "old_logit_bias_multiseed": {"overall": 0.1714, "old": 0.2016, "new": 0.0508, "forgetting": 0.3175},
}


def main() -> int:
    """汇总 seed7 结果，并判断是否扩三种子。"""
    parser = argparse.ArgumentParser(description="LoRa BN 重校准 seed7 报告。")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    metrics = extract_round_metrics(save_dir)
    r3 = next(row for row in metrics["rounds"] if row["round"] == "R3")
    cluster_ok = all(int(row["cluster_count"]) == 5 for row in metrics["rounds"])
    overall_improved = float(r3["overall_acc"]) > REFERENCES["grouped_hybrid"]["overall"]
    old_not_worse = float(r3["old_acc"]) >= REFERENCES["selected_radcil"]["old"]
    new_preserved = float(r3["new_acc"]) >= REFERENCES["selected_radcil"]["new"] - 0.02
    expand_multiseed = bool(cluster_ok and overall_improved and old_not_worse and new_preserved)

    lines = [
        "# 阶段 5 LoRa BatchNorm 重校准 seed7 诊断报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 目标：验证 LoRa 低分是否主要来自跨天 BatchNorm 统计漂移。",
        "- 边界：BN 只用 replay memory 和当前 discovery/enrollment 样本重校准；IQ_8-10 held-out eval 不参与统计估计或选参。",
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
    lines.extend([
        "",
        "## 判定",
        "",
        f"- 三轮目标簇数：{'通过' if cluster_ok else '未通过'}。",
        f"- Overall 是否超过 grouped hybrid：{'通过' if overall_improved else '未通过'}。",
        f"- Old 是否不低于 selected RADCIL：{'通过' if old_not_worse else '未通过'}。",
        f"- New 是否基本保持 selected RADCIL：{'通过' if new_preserved else '未通过'}。",
        f"- 是否扩展 seed13/31：{'是' if expand_multiseed else '否'}。",
        "",
    ])

    summary = {
        "job_id": args.job_id,
        "save_dir": str(save_dir),
        "metrics": metrics,
        "references": REFERENCES,
        "cluster_count_gate_passed": cluster_ok,
        "overall_improved_over_grouped_hybrid": overall_improved,
        "old_not_worse_than_selected_radcil": old_not_worse,
        "new_preserved_vs_selected_radcil": new_preserved,
        "expand_multiseed": expand_multiseed,
    }
    output_path = Path(args.output)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"LoRa BN 重校准报告：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
