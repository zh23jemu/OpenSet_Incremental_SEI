"""汇总 RADCIL + iCaRL 低置信回退 seed7 结果并执行门槛判断。"""

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


def read_r3_metrics(csv_path: Path) -> dict[str, float]:
    """读取 iCaRL fallback 的唯一 R3 指标行。"""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("Method") == "MV-ACC-CIL + iCaRL-fallback" and row.get("Stage") == "After R3"
        ]
    if len(rows) != 1:
        raise ValueError(f"期望唯一 iCaRL fallback R3 结果，实际找到 {len(rows)} 行：{csv_path}")
    return {key: float(rows[0][column]) for key, column in METRIC_COLUMNS.items()}


def read_baseline_r3(summary_path: Path) -> dict[str, float]:
    """从阶段 2 seed7 汇总读取同协议网络后端基线。"""
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [row for row in data["metrics"]["rounds"] if row.get("round") == "R3"]
    if len(rows) != 1:
        raise ValueError(f"阶段 2 汇总缺少唯一 R3 行：{summary_path}")
    return {key: float(rows[0][key]) for key in METRIC_COLUMNS}


def read_calibration_rows(save_dir: Path) -> list[dict[str, str]]:
    """读取 Day1 validation 阈值校准轨迹；缺失时返回空列表，便于旧结果兼容。"""
    path = save_dir / "icarl_fallback_iq7_calibration.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_summary(plan_path: Path, baseline_path: Path) -> dict:
    """计算 fallback 与原 RADCIL seed7 的差异及是否扩展三种子。"""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    save_dir = Path(plan["run"]["save_dir"])
    fallback = read_r3_metrics(save_dir / "incremental_results.csv")
    baseline = read_baseline_r3(baseline_path)
    delta = {key: fallback[key] - baseline[key] for key in METRIC_COLUMNS}
    old_or_forgetting_improved = (
        delta["old_acc"] > 0.0
        or delta["forgetting_rate"] < 0.0
    )
    expand = fallback["overall_acc"] >= baseline["overall_acc"] and old_or_forgetting_improved
    return {
        "schema_version": "stage3_wisig_icarl_fallback_seed7_summary_v1",
        "plan": plan,
        "baseline_summary": str(baseline_path),
        "fallback_r3": fallback,
        "baseline_r3": baseline,
        "delta_fallback_minus_baseline": delta,
        "calibration_rows": read_calibration_rows(save_dir),
        "old_or_forgetting_improved": bool(old_or_forgetting_improved),
        "expand_to_three_seeds": bool(expand),
        "decision": (
            "达到门槛，可进入三种子正式验证。"
            if expand
            else "未达到门槛，保留为负消融并停止该机制。"
        ),
    }


def render_report(summary: dict) -> str:
    """渲染 Markdown 报告，突出是否真正收敛旧类风险。"""
    fallback = summary["fallback_r3"]
    baseline = summary["baseline_r3"]
    delta = summary["delta_fallback_minus_baseline"]
    lines = [
        "# 阶段 3 WiSig RADCIL + iCaRL Fallback Seed7 报告",
        "",
        "| 方法 | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| MV-ACC-CIL + iCaRL-fallback | {fallback['overall_acc']:.4f} | {fallback['old_acc']:.4f} | {fallback['new_acc']:.4f} | {fallback['forgetting_rate']:.4f} | {fallback['macro_f1']:.4f} |",
        f"| MV-ACC-CIL seed7 | {baseline['overall_acc']:.4f} | {baseline['old_acc']:.4f} | {baseline['new_acc']:.4f} | {baseline['forgetting_rate']:.4f} | {baseline['macro_f1']:.4f} |",
        f"| 差值 | {delta['overall_acc']:+.4f} | {delta['old_acc']:+.4f} | {delta['new_acc']:+.4f} | {delta['forgetting_rate']:+.4f} | {delta['macro_f1']:+.4f} |",
        "",
        f"- 决策：{summary['decision']}",
        "- 协议边界：fallback 原型仅由训练/伪标签 replay 记忆构建，阈值只由 Day1 validation 选择，held-out 真值只用于事后评估。",
        "",
    ]
    if summary["calibration_rows"]:
        lines.extend([
            "## Day1 Validation 阈值校准",
            "",
            "| Stage | Threshold | Validation Accuracy |",
            "| --- | ---: | ---: |",
        ])
        for row in summary["calibration_rows"]:
            lines.append(
                f"| {row.get('Stage', '')} | {float(row.get('Fallback Threshold', 0.0)):.2f} | "
                f"{float(row.get('Validation Accuracy', 0.0)):.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 WiSig iCaRL fallback seed7 结果。")
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
    print(f"WiSig iCaRL fallback seed7 报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
