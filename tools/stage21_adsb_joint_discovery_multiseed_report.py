"""汇总 ADS-B Stage 21 联合 discovery-CIL 三种子结果。

该报告只读取每个 seed 已经生成的 ``per_round_summary_results.csv``，
不重新计算聚类或增量指标，也不读取 held-out 真值参与任何训练决策。
这样可以把 Slurm 运行和结果汇总分开，避免报告脚本意外改变实验行为。
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    """读取单个 seed 的轮次汇总，并在文件缺失时给出明确错误。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"结果文件为空：{path}")
    return rows


def _number(row: dict[str, str], *keys: str) -> float:
    """兼容不同阶段报告中的列名，缺失指标统一返回 NaN。"""

    for key in keys:
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
    return float("nan")


def _last_round(rows: list[dict[str, str]]) -> dict[str, str]:
    """优先取 R3；若旧格式没有 R3，则退回最后一行。"""

    for row in rows:
        if str(row.get("Round", "")).upper() == "R3":
            return row
    return rows[-1]


def _mean_std(values: list[float]) -> tuple[float, float]:
    """计算总体均值和标准差，保持与项目其它多种子报告一致。"""

    valid = [value for value in values if not math.isnan(value)]
    if not valid:
        return float("nan"), float("nan")
    mean = sum(valid) / len(valid)
    variance = sum((value - mean) ** 2 for value in valid) / len(valid)
    return mean, math.sqrt(variance)


def _fmt(value: float) -> str:
    """统一客户报告中的四位小数格式。"""

    return "nan" if math.isnan(value) else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 ADS-B Stage 21 联合 discovery-CIL 三种子结果。")
    parser.add_argument("--root", default="results/stage21", help="Stage 21 结果根目录。")
    parser.add_argument("--job-id", required=True, help="当前 Slurm job id。")
    parser.add_argument("--seeds", default="7,13,31", help="逗号分隔的 seed 列表。")
    parser.add_argument("--output", required=True, help="Markdown 报告输出路径。")
    parser.add_argument("--summary-json", default=None, help="可选的机器可读汇总 JSON 路径。")
    args = parser.parse_args()

    root = Path(args.root)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    rows: list[dict[str, float | int]] = []
    for seed in seeds:
        save_dir = root / f"adsb_joint_discovery_cil_seed{seed}_{args.job_id}"
        row = _last_round(_read_rows(save_dir / "per_round_summary_results.csv"))
        rows.append(
            {
                "seed": seed,
                "overall": _number(row, "Overall Acc", "Overall"),
                "old": _number(row, "Old Acc", "Old"),
                "new": _number(row, "New Acc", "New"),
                "forgetting": _number(row, "Forgetting Rate", "Forgetting"),
                "macro_f1": _number(row, "Macro F1"),
                "cluster_count": _number(row, "Cluster Count", "Final Cluster Count"),
                "hungarian": _number(row, "Hungarian Acc", "Hungarian"),
            }
        )

    lines = [
        "# Stage 21 ADS-B 联合 discovery-CIL 三种子报告",
        "",
        "组合：GPCC 固定目标簇数 + 每轮第一次 CIL + Student 重新发现 + 无标签原型 Hungarian 对齐 + 第二次 CIL。",
        "",
        "| Seed | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 | 簇数 | Hungarian |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {_fmt(float(row['overall']))} | {_fmt(float(row['old']))} | "
            f"{_fmt(float(row['new']))} | {_fmt(float(row['forgetting']))} | "
            f"{_fmt(float(row['macro_f1']))} | {_fmt(float(row['cluster_count']))} | "
            f"{_fmt(float(row['hungarian']))} |"
        )

    lines.extend(["", "## 三种子均值 ± 标准差", ""])
    for key, label in (
        ("overall", "R3 Overall"),
        ("old", "R3 Old"),
        ("new", "R3 New"),
        ("forgetting", "Forgetting"),
        ("macro_f1", "Macro F1"),
        ("cluster_count", "R3 簇数"),
        ("hungarian", "R3 Hungarian"),
    ):
        mean, std = _mean_std([float(row[key]) for row in rows])
        lines.append(f"- {label}: {_fmt(mean)} ± {_fmt(std)}")

    baseline = 0.5057
    overall_mean, _ = _mean_std([float(row["overall"]) for row in rows])
    old_mean, _ = _mean_std([float(row["old"]) for row in rows])
    new_mean, _ = _mean_std([float(row["new"]) for row in rows])
    if overall_mean > baseline and old_mean >= 0.4850 - 0.02 and new_mean >= 0.5583 - 0.03:
        verdict = "seed7 已通过，但三种子仍需确认是否稳定超过强对照后再锁定。"
    else:
        verdict = "未达到三种子预注册门槛；不继续做相邻小机制搜索，应转更大的联合表征设计或如实收口。"
    lines.extend(
        [
            "",
            f"判定基准：seed7 强对照 R3 Overall={baseline:.4f}、Old=0.4850、New=0.5583。",
            f"当前判定：{verdict}",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.summary_json:
        import json

        summary = {
            "job_id": str(args.job_id),
            "seeds": rows,
            "mean_std": {
                key: {"mean": _mean_std([float(row[key]) for row in rows])[0], "std": _mean_std([float(row[key]) for row in rows])[1]}
                for key in ("overall", "old", "new", "forgetting", "macro_f1", "cluster_count", "hungarian")
            },
            "baseline": {"overall": baseline, "old": 0.4850, "new": 0.5583},
            "verdict": verdict,
        }
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
