"""汇总阶段 3 WiSig 正式 CIL baseline 三种子结果。

本脚本读取开启 CIL baseline 后生成的每个 seed 输出目录，分别汇总：

1. 共享 MV-ACC 伪标签的增量后端 baseline，用于隔离后端遗忘控制能力；
2. Deep-HDBSCAN + 常规 CIL 的端到端 baseline，用于完整系统对照；
3. 主方法 MV-ACC-CIL 的同目录结果，用于确认 baseline job 与主配置一致。

脚本只做结果聚合，不重新训练，也不使用评估真值参与任何模型选择。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SEEDS = (7, 13, 31)
R3_STAGE = "After R3"
METRICS = (
    ("Overall Acc", "overall"),
    ("Old Acc", "old"),
    ("New Acc", "new"),
    ("Forgetting Rate", "forgetting"),
    ("Macro F1", "macro_f1"),
)


def parse_seeds(text: str) -> tuple[int, ...]:
    """解析逗号分隔 seed，并保持用户给定顺序。"""
    seeds: list[int] = []
    for item in text.split(","):
        value = item.strip()
        if not value:
            continue
        seed = int(value)
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise ValueError("至少需要一个 seed。")
    return tuple(seeds)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """读取 CSV 文件；baseline 正式报告缺文件时应直接失败。"""
    if not path.exists():
        raise FileNotFoundError(f"缺少 baseline 结果文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | None) -> float | None:
    """将 CSV 单元格转换成浮点数，空值保留为 None。"""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value: float | None) -> str:
    """Markdown 表格统一数值格式。"""
    if value is None or math.isnan(float(value)):
        return "NA"
    return f"{float(value):.4f}"


def mean_std(values: list[float | None]) -> dict[str, float | None]:
    """计算均值和样本标准差。"""
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not clean:
        return {"mean": None, "std": None}
    if len(clean) == 1:
        return {"mean": clean[0], "std": 0.0}
    return {"mean": statistics.mean(clean), "std": statistics.stdev(clean)}


def collect_r3_rows(path: Path, seed: int, table_name: str) -> list[dict[str, Any]]:
    """从一个结果 CSV 中提取 R3 行。

    不同 CSV 的 Method 命名不同，但 Stage/R3 和核心指标列保持一致。这里
    统一转成内部结构，后续即可按 table/method 聚合。
    """
    rows: list[dict[str, Any]] = []
    for row in read_csv_rows(path):
        if str(row.get("Stage", "")) != R3_STAGE:
            continue
        item: dict[str, Any] = {
            "seed": seed,
            "table": table_name,
            "method": row.get("Method", "unknown"),
            "source_csv": str(path),
        }
        for csv_key, key in METRICS:
            item[key] = to_float(row.get(csv_key))
        rows.append(item)
    return rows


def collect_seed(seed: int, save_dir: Path) -> list[dict[str, Any]]:
    """收集单个 seed 的主方法和 baseline R3 结果。"""
    rows: list[dict[str, Any]] = []
    rows.extend(collect_r3_rows(save_dir / "incremental_results.csv", seed, "main_method"))
    rows.extend(collect_r3_rows(save_dir / "shared_discovery_baseline_results.csv", seed, "shared_discovery_backend"))
    rows.extend(collect_r3_rows(save_dir / "end_to_end_baseline_results.csv", seed, "end_to_end_deep_hdbscan"))
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按表类型和方法名聚合三种子 R3 均值/标准差。"""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["table"]), str(row["method"])), []).append(row)

    summary: list[dict[str, Any]] = []
    for (table, method), items in sorted(groups.items()):
        item: dict[str, Any] = {
            "table": table,
            "method": method,
            "seed_count": len(items),
        }
        for _, key in METRICS:
            stats = mean_std([row.get(key) for row in items])
            item[f"{key}_mean"] = stats["mean"]
            item[f"{key}_std"] = stats["std"]
        summary.append(item)
    return summary


def table_rows(summary: list[dict[str, Any]], table_name: str) -> list[dict[str, Any]]:
    """取出指定 baseline 表的聚合结果，并按 Overall 均值降序排列。"""
    rows = [row for row in summary if row["table"] == table_name]
    return sorted(rows, key=lambda row: float(row.get("overall_mean") or -1.0), reverse=True)


def render_metric_table(rows: list[dict[str, Any]]) -> list[str]:
    """渲染 R3 均值/标准差表。"""
    lines = [
        "| 方法 | Seeds | Overall | Old | New | Forgetting | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| `{method}` | {seed_count} | {overall}±{overall_std} | {old}±{old_std} | {new}±{new_std} | {forgetting}±{forgetting_std} | {macro_f1}±{macro_f1_std} |".format(
                method=row["method"],
                seed_count=row["seed_count"],
                overall=fmt(row.get("overall_mean")),
                overall_std=fmt(row.get("overall_std")),
                old=fmt(row.get("old_mean")),
                old_std=fmt(row.get("old_std")),
                new=fmt(row.get("new_mean")),
                new_std=fmt(row.get("new_std")),
                forgetting=fmt(row.get("forgetting_mean")),
                forgetting_std=fmt(row.get("forgetting_std")),
                macro_f1=fmt(row.get("macro_f1_mean")),
                macro_f1_std=fmt(row.get("macro_f1_std")),
            )
        )
    return lines


def find_method(summary: list[dict[str, Any]], table: str, method: str) -> dict[str, Any] | None:
    """按表名和方法名查找聚合行。"""
    for row in summary:
        if row["table"] == table and row["method"] == method:
            return row
    return None


def build_report(job_id: str, seeds: tuple[int, ...], output_prefix: str, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> str:
    """生成正式 baseline Markdown 报告。"""
    main = find_method(summary, "main_method", "MV-ACC-CIL")
    shared = table_rows(summary, "shared_discovery_backend")
    end_to_end = table_rows(summary, "end_to_end_deep_hdbscan")
    best_shared = shared[0] if shared else None
    best_e2e = end_to_end[0] if end_to_end else None

    lines = [
        "# 阶段 3 WiSig 正式 CIL Baseline 三种子报告",
        "",
        f"- 生成时间 UTC：{datetime.now(timezone.utc).isoformat()}",
        f"- Slurm Job：`{job_id}`",
        f"- Seeds：`{', '.join(str(seed) for seed in seeds)}`",
        f"- 输出前缀：`{output_prefix}`",
        "- 协议：WiSig strict 10+10x3，MV-ACC + `ratio_2p0_replay_3p0` 主配置训练预算保持不变，仅开启内置 CIL baseline。",
        "- 评估口径：所有表均报告 R3 三种子均值和标准差；共享发现表隔离后端，端到端表同时评估 Deep-HDBSCAN 发现和 CIL 后端。",
        "",
        "## 主方法一致性检查",
        "",
    ]
    if main is not None:
        lines.extend(render_metric_table([main]))
    else:
        lines.append("- 未找到 `MV-ACC-CIL` 主方法 R3 行。")

    lines.extend(
        [
            "",
            "## 共享 MV-ACC 伪标签的增量后端 Baseline",
            "",
            *render_metric_table(shared),
            "",
            "## Deep-HDBSCAN 端到端 Baseline",
            "",
            *render_metric_table(end_to_end),
            "",
            "## 当前判断",
            "",
        ]
    )

    if main is not None and best_shared is not None:
        delta = float(main["overall_mean"]) - float(best_shared["overall_mean"])
        relation = "高" if delta >= 0 else "低"
        lines.append(
            f"- 在共享发现后端对照中，主方法 R3 Overall 比最佳 baseline `{best_shared['method']}` {relation} `{abs(delta):.4f}`。"
        )
    if main is not None and best_e2e is not None:
        delta = float(main["overall_mean"]) - float(best_e2e["overall_mean"])
        relation = "高" if delta >= 0 else "低"
        lines.append(
            f"- 在 Deep-HDBSCAN 端到端对照中，主方法 R3 Overall 比最佳 baseline `{best_e2e['method']}` {relation} `{abs(delta):.4f}`。"
        )
    if main is not None and best_shared is not None and float(main["overall_mean"]) < float(best_shared["overall_mean"]):
        lines.append(
            "- 共享发现表说明当前网络式 RADCIL 后端不是后端上限；后续应把 DOI-style/iCaRL/TPCIL-style 作为强后端候选或混合后端消融，而不是只强调当前网络后端。"
        )
    lines.extend(
        [
            "- 该报告用于补齐 WiSig 阶段 3 的核心 baseline 与增量 baseline 表；IGCD-minimal/SimGCD-style 仍需单独作为 strict/minimal 前端适配对照标注。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总阶段 3 WiSig 正式 CIL baseline 三种子结果。")
    parser.add_argument("--job-id", required=True, help="Slurm Job ID。")
    parser.add_argument("--seeds", default="7,13,31", help="逗号分隔 seed 列表。")
    parser.add_argument("--output-prefix", default="results/stage3/wisig_cil_baselines", help="baseline 输出目录前缀。")
    parser.add_argument("--output", required=True, help="Markdown 报告输出路径。")
    parser.add_argument("--summary-json", required=True, help="JSON 汇总输出路径。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        save_dir = Path(f"{args.output_prefix}_ratio_2p0_replay_3p0_seed{seed}_{args.job_id}")
        rows.extend(collect_seed(seed, save_dir))
    summary = aggregate(rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(args.job_id, seeds, args.output_prefix, rows, summary), encoding="utf-8")

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"job_id": args.job_id, "seeds": list(seeds), "per_seed_r3": rows, "summary": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"阶段 3 WiSig CIL baseline 三种子报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
