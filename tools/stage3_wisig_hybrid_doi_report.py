"""汇总 hybrid RADCIL + DOI-memory seed 7 结果并执行扩展门槛判断。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRIC_COLUMNS = {
    "overall_acc": "Overall Acc",
    "old_acc": "Old Acc",
    "new_acc": "New Acc",
    "forgetting_rate": "Forgetting Rate",
    "macro_f1": "Macro F1",
}

# Job 44453416 在共享 MV-ACC 发现条件下得到的 DOI-style 三种子参考。
# 允许 0.02 的绝对容差，避免用单种子偶然波动决定是否扩展正式三种子。
DOI_REFERENCE = {"old_acc": 0.5721, "forgetting_rate": 0.1959}
REFERENCE_TOLERANCE = 0.02


def read_r3_metrics(csv_path: Path) -> dict[str, float]:
    """读取混合后端 R3 指标，并验证方法名和轮次唯一。"""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("Method") == "MV-ACC-CIL + DOI-memory" and row.get("Stage") == "After R3"
        ]
    if len(rows) != 1:
        raise ValueError(f"期望唯一 hybrid DOI R3 结果，实际找到 {len(rows)} 行：{csv_path}")
    return {key: float(rows[0][column]) for key, column in METRIC_COLUMNS.items()}


def read_baseline_r3(summary_path: Path) -> dict[str, float]:
    """从阶段 2 seed7 汇总读取同协议网络后端基线。"""
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [row for row in data["metrics"]["rounds"] if row.get("round") == "R3"]
    if len(rows) != 1:
        raise ValueError(f"阶段 2 汇总缺少唯一 R3 行：{summary_path}")
    return {key: float(rows[0][key]) for key in METRIC_COLUMNS}


def build_summary(plan_path: Path, baseline_path: Path) -> dict:
    """计算 hybrid 与原 RADCIL seed7 的差异及是否扩展三种子。"""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    incremental_path = Path(plan["run"]["save_dir"]) / "incremental_results.csv"
    hybrid = read_r3_metrics(incremental_path)
    baseline = read_baseline_r3(baseline_path)
    delta = {key: hybrid[key] - baseline[key] for key in METRIC_COLUMNS}
    near_reference = (
        hybrid["old_acc"] >= DOI_REFERENCE["old_acc"] - REFERENCE_TOLERANCE
        or hybrid["forgetting_rate"] <= DOI_REFERENCE["forgetting_rate"] + REFERENCE_TOLERANCE
    )
    expand = hybrid["overall_acc"] >= baseline["overall_acc"] and near_reference
    return {
        "schema_version": "stage3_wisig_hybrid_doi_seed7_summary_v1",
        "plan": plan,
        "baseline_summary": str(baseline_path),
        "hybrid_r3": hybrid,
        "baseline_r3": baseline,
        "delta_hybrid_minus_baseline": delta,
        "doi_reference": DOI_REFERENCE,
        "doi_reference_tolerance": REFERENCE_TOLERANCE,
        "near_doi_reference": bool(near_reference),
        "expand_to_three_seeds": bool(expand),
        "decision": (
            "达到门槛，扩展 seed 7/13/31。"
            if expand
            else "未达到门槛，保留为负消融并停止三种子扩展。"
        ),
    }


def render_report(summary: dict) -> str:
    """渲染便于客户和论文复盘的 Markdown 报告。"""
    hybrid = summary["hybrid_r3"]
    baseline = summary["baseline_r3"]
    delta = summary["delta_hybrid_minus_baseline"]
    lines = [
        "# 阶段 3 WiSig Hybrid RADCIL + DOI-memory Seed7 报告",
        "",
        "| 方法 | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| MV-ACC-CIL + DOI-memory | {hybrid['overall_acc']:.4f} | {hybrid['old_acc']:.4f} | {hybrid['new_acc']:.4f} | {hybrid['forgetting_rate']:.4f} | {hybrid['macro_f1']:.4f} |",
        f"| MV-ACC-CIL seed7 | {baseline['overall_acc']:.4f} | {baseline['old_acc']:.4f} | {baseline['new_acc']:.4f} | {baseline['forgetting_rate']:.4f} | {baseline['macro_f1']:.4f} |",
        f"| 差值 | {delta['overall_acc']:+.4f} | {delta['old_acc']:+.4f} | {delta['new_acc']:+.4f} | {delta['forgetting_rate']:+.4f} | {delta['macro_f1']:+.4f} |",
        "",
        f"- 决策：{summary['decision']}",
        "- 协议边界：融合原型仅由训练/伪标签 replay 记忆构建，held-out 真值只用于事后评估。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 WiSig hybrid DOI seed7 结果。")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--baseline-summary", default="results/stage2/stage2_wisig_main_single_seed_summary_44433946.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_summary(Path(args.plan), Path(args.baseline_summary))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(summary), encoding="utf-8")
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"WiSig hybrid DOI seed7 报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
