"""汇总 WiSig RADCIL + iCaRL fallback 三种子结果并对比主方法。

报告器只读取已经完成的严格协议输出，不重新训练，也不使用任何 held-out
真值做参数选择。它的职责是把 fallback 与阶段 3 主方法三种子均值放到
同一套 R1/R2/R3 指标里，明确是否应升级为主后端候选。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


METRICS = (
    ("overall_acc", "Overall Acc", "Overall"),
    ("old_acc", "Old Acc", "Old"),
    ("new_acc", "New Acc", "New"),
    ("forgetting_rate", "Forgetting Rate", "Forgetting"),
    ("macro_f1", "Macro F1", "Macro F1"),
)
ROUNDS = ("R1", "R2", "R3")


def fmt(value: float) -> str:
    """统一报告数值格式。"""
    return f"{value:.4f}"


def read_incremental_rows(save_dir: Path, seed: int) -> list[dict]:
    """读取单 seed 的 R1/R2/R3 incremental 结果。"""
    path = save_dir / "incremental_results.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("Method") == "MV-ACC-CIL + iCaRL-fallback"
            and row.get("Stage", "").startswith("After R")
        ]
    if len(rows) != 3:
        raise ValueError(f"期望 seed {seed} 有 3 行 R1/R2/R3 结果，实际为 {len(rows)}：{path}")
    parsed = []
    for row in rows:
        parsed.append({
            "seed": int(seed),
            "round": row["Stage"].replace("After ", ""),
            "save_dir": str(save_dir),
            **{key: float(row[column]) for key, column, _ in METRICS},
        })
    return parsed


def summarize(rows: list[dict]) -> list[dict]:
    """按轮次计算三种子均值和样本标准差。"""
    summary = []
    for round_name in ROUNDS:
        sub = [row for row in rows if row["round"] == round_name]
        if not sub:
            raise ValueError(f"缺少 {round_name} 结果。")
        item = {"round": round_name, "seed_count": len(sub)}
        for key, _, _ in METRICS:
            values = [float(row[key]) for row in sub]
            item[f"{key}_mean"] = mean(values)
            item[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
        summary.append(item)
    return summary


def baseline_rounds(path: Path) -> dict[str, dict]:
    """读取阶段 3 主方法三种子 summary，按轮次索引。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["round"]: row for row in payload["summary"]}


def build_summary(plan_path: Path, baseline_path: Path) -> dict:
    """构建 JSON 摘要和正式门槛判断。"""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for run in plan["runs"]:
        rows.extend(read_incremental_rows(Path(run["save_dir"]), int(run["seed"])))
    fallback_summary = summarize(rows)
    fallback_by_round = {row["round"]: row for row in fallback_summary}
    baseline_by_round = baseline_rounds(baseline_path)
    r3_fb = fallback_by_round["R3"]
    r3_base = baseline_by_round["R3"]
    delta = {
        key: float(r3_fb[f"{key}_mean"]) - float(r3_base[f"{key}_mean"])
        for key, _, _ in METRICS
    }
    old_or_forgetting_improved = delta["old_acc"] > 0.0 or delta["forgetting_rate"] < 0.0
    promote = delta["overall_acc"] >= 0.0 and old_or_forgetting_improved
    return {
        "schema_version": "stage3_wisig_icarl_fallback_multiseed_summary_v1",
        "plan": plan,
        "baseline_summary": str(baseline_path),
        "per_seed_rows": rows,
        "summary": fallback_summary,
        "baseline_r3": r3_base,
        "delta_r3_fallback_minus_baseline": delta,
        "old_or_forgetting_improved": bool(old_or_forgetting_improved),
        "promote_to_main_backend_candidate": bool(promote),
        "decision": (
            "三种子通过门槛，可作为 WiSig 后端上限风险的主收敛候选。"
            if promote
            else "三种子未通过门槛，保留为负消融或局部 seed7 收益。"
        ),
    }


def render_report(summary: dict) -> str:
    """渲染 Markdown 汇总报告。"""
    by_round = {row["round"]: row for row in summary["summary"]}
    base_r3 = summary["baseline_r3"]
    delta = summary["delta_r3_fallback_minus_baseline"]
    lines = [
        "# 阶段 3 WiSig RADCIL + iCaRL Fallback 三种子报告",
        "",
        f"- 决策：{summary['decision']}",
        "- 边界：同 WiSig strict 10+10x3、同 MV-ACC 前端、同 RADCIL ratio_2p0_replay_3p0 训练预算，只新增低置信 exemplar fallback。",
        "- 校准：fallback 阈值只用 Day1 validation；held-out evaluation 真值只用于最终事后评估。",
        "",
        "## 三种子均值",
        "",
        "| Round | Overall | Old | New | Forgetting | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for round_name in ROUNDS:
        row = by_round[round_name]
        lines.append(
            f"| {round_name} | "
            f"{fmt(row['overall_acc_mean'])}±{fmt(row['overall_acc_std'])} | "
            f"{fmt(row['old_acc_mean'])}±{fmt(row['old_acc_std'])} | "
            f"{fmt(row['new_acc_mean'])}±{fmt(row['new_acc_std'])} | "
            f"{fmt(row['forgetting_rate_mean'])}±{fmt(row['forgetting_rate_std'])} | "
            f"{fmt(row['macro_f1_mean'])}±{fmt(row['macro_f1_std'])} |"
        )
    lines.extend([
        "",
        "## R3 对比主方法",
        "",
        "| 配置 | Overall | Old | New | Forgetting | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| MV-ACC-CIL + iCaRL-fallback | "
        f"{fmt(by_round['R3']['overall_acc_mean'])} | {fmt(by_round['R3']['old_acc_mean'])} | "
        f"{fmt(by_round['R3']['new_acc_mean'])} | {fmt(by_round['R3']['forgetting_rate_mean'])} | "
        f"{fmt(by_round['R3']['macro_f1_mean'])} |",
        "| MV-ACC-CIL main | "
        f"{fmt(float(base_r3['overall_acc_mean']))} | {fmt(float(base_r3['old_acc_mean']))} | "
        f"{fmt(float(base_r3['new_acc_mean']))} | {fmt(float(base_r3['forgetting_rate_mean']))} | "
        f"{fmt(float(base_r3['macro_f1_mean']))} |",
        "| 差值 | "
        f"{delta['overall_acc']:+.4f} | {delta['old_acc']:+.4f} | {delta['new_acc']:+.4f} | "
        f"{delta['forgetting_rate']:+.4f} | {delta['macro_f1']:+.4f} |",
        "",
        "## 单种子 R3",
        "",
        "| Seed | Overall | Old | New | Forgetting | Macro F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in summary["per_seed_rows"]:
        if row["round"] != "R3":
            continue
        lines.append(
            f"| {row['seed']} | {fmt(row['overall_acc'])} | {fmt(row['old_acc'])} | "
            f"{fmt(row['new_acc'])} | {fmt(row['forgetting_rate'])} | {fmt(row['macro_f1'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 WiSig iCaRL fallback 三种子结果。")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--baseline-summary", default="results/stage3/stage3_wisig_main_multiseed_summary_44440345.json")
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
    print(f"WiSig iCaRL fallback 三种子报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
