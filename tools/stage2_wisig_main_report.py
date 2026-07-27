"""生成阶段 2 WiSig 单种子主流程报告。

该脚本在 Slurm 实验结束后读取 `clustering_results.csv`、
`incremental_results.csv` 和 `per_round_summary_results.csv`，提取 R1/R2/R3
关键指标并写成 Markdown。它只做结果汇总，不重新计算训练或聚类逻辑。
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROUND_ORDER = ("R1", "R2", "R3")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """读取 CSV 行；缺失文件直接报错，避免生成看似完整但证据不足的报告。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少结果文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | None) -> float | None:
    """把 CSV 单元格转换为浮点数；空值保留为 None，便于报告标记缺失。"""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value: Any) -> str:
    """统一格式化表格数值，缺失值显示为 `NA`。"""
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def detect_round(row: dict[str, str]) -> str | None:
    """从 Round 或 Stage 字段识别 R1/R2/R3。

    发现结果表通常直接使用 `Round=R1`，增量结果表则可能使用
    `Stage=After R1`。这里同时兼容两种写法，避免 Old Acc 和 Macro F1
    这类只在增量表中的指标被误判为缺失。
    """
    text = " ".join([row.get("Round", ""), row.get("Stage", "")])
    for round_name in ROUND_ORDER:
        if round_name in text:
            return round_name
    return None


def rows_by_round(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """按轮次索引结果行，只保留 R1/R2/R3。"""
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        round_name = detect_round(row)
        if round_name in ROUND_ORDER:
            indexed[round_name] = row
    return indexed


def extract_round_metrics(save_dir: Path) -> dict[str, Any]:
    """从实验输出目录提取 discovery 与 incremental 指标。"""
    clustering = rows_by_round(read_csv_rows(save_dir / "clustering_results.csv"))
    incremental = rows_by_round(read_csv_rows(save_dir / "incremental_results.csv"))
    summary = rows_by_round(read_csv_rows(save_dir / "per_round_summary_results.csv"))

    rounds: list[dict[str, Any]] = []
    for round_name in ROUND_ORDER:
        cluster_row = clustering.get(round_name, {})
        inc_row = incremental.get(round_name, {})
        summary_row = summary.get(round_name, {})
        rounds.append(
            {
                "round": round_name,
                "cluster_count": to_float(summary_row.get("Final Cluster Count") or cluster_row.get("Final Cluster Count")),
                "nmi": to_float(summary_row.get("NMI") or cluster_row.get("NMI")),
                "ari": to_float(summary_row.get("ARI") or cluster_row.get("ARI")),
                "purity": to_float(summary_row.get("Purity") or cluster_row.get("Purity")),
                "hungarian_acc": to_float(summary_row.get("Hungarian Acc") or cluster_row.get("Hungarian Acc")),
                "overall_acc": to_float(summary_row.get("Overall Acc") or inc_row.get("Overall Acc")),
                "old_acc": to_float(inc_row.get("Old Acc")),
                "new_acc": to_float(summary_row.get("New Acc") or inc_row.get("New Acc")),
                "forgetting_rate": to_float(summary_row.get("Forgetting Rate") or inc_row.get("Forgetting Rate")),
                "macro_f1": to_float(inc_row.get("Macro F1")),
            }
        )
    return {"rounds": rounds}


def load_plan(path: Path | None) -> dict[str, Any]:
    """读取计划 JSON；没有提供时返回空字典，报告仍可基于结果生成。"""
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"缺少计划文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_markdown(save_dir: Path, plan: dict[str, Any], metrics: dict[str, Any], job_id: str) -> str:
    """构造客户可读的阶段 2 单种子 Markdown 报告。"""
    config = plan.get("locked_main_config", {})
    lines = [
        "# 阶段 2 WiSig RADCIL 单种子主流程报告",
        "",
        f"- 生成时间 UTC：{datetime.now(timezone.utc).isoformat()}",
        f"- Slurm Job：`{job_id}`",
        f"- 输出目录：`{save_dir.as_posix()}`",
        f"- 主前端：`{config.get('frontend', 'MV-ACC')}`",
        f"- 主后端：`{config.get('variant', 'ratio_2p0_replay_3p0')}`",
        f"- Seed：`{config.get('seed', 7)}`",
        "- 协议：WiSig strict 10+10x3，沿用阶段 0 的 60/10/30 隔离边界。",
        "",
        "## R1/R2/R3 核心指标",
        "",
        "| 轮次 | Cluster Count | NMI | ARI | Purity | Hungarian Acc | Overall | Old | New | Forgetting | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in metrics["rounds"]:
        lines.append(
            "| {round} | {cluster_count} | {nmi} | {ari} | {purity} | {hungarian_acc} | {overall_acc} | {old_acc} | {new_acc} | {forgetting_rate} | {macro_f1} |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )

    r3 = next((row for row in metrics["rounds"] if row["round"] == "R3"), {})
    lines.extend(
        [
            "",
            "## 初步判断",
            "",
            f"- R3 Overall：`{fmt(r3.get('overall_acc'))}`，Old：`{fmt(r3.get('old_acc'))}`，New：`{fmt(r3.get('new_acc'))}`，Forgetting：`{fmt(r3.get('forgetting_rate'))}`。",
            "- 这是阶段 2 单种子完整三轮主流程结果，可用于判断流程是否打通；正式主结果仍需要后续三种子均值和标准差。",
            "- 若 R3 旧类保持仍弱，下一步优先在该主配置上继续做 old:new ratio、replay 权重或类别头校准的小范围补充，而不是重新选择较弱发现前端。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成阶段 2 WiSig 单种子主流程报告。")
    parser.add_argument("--save-dir", required=True, help="实验输出目录。")
    parser.add_argument("--plan", default=None, help="对应计划 JSON 路径。")
    parser.add_argument("--job-id", default="manual", help="Slurm Job ID。")
    parser.add_argument("--output", required=True, help="Markdown 报告输出路径。")
    parser.add_argument("--summary-json", default=None, help="可选 JSON 指标摘要输出路径。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    save_dir = Path(args.save_dir)
    plan = load_plan(Path(args.plan) if args.plan else None)
    metrics = extract_round_metrics(save_dir)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(save_dir, plan, metrics, args.job_id), encoding="utf-8")

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps({"plan": plan, "metrics": metrics}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"阶段 2 WiSig 单种子报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
