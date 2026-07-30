"""汇总 Stage 17 ADS-B 高置信伪标签注册 seed7 矩阵结果。

该阶段验证“低置信 discovery 样本是否正在污染新增分类头”。报告只读取
incremental/clustering CSV 小结果，不读取模型权重或 replay memory。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


BASELINE_R3 = {
    "Overall Acc": 0.5057,
    "Old Acc": 0.4850,
    "New Acc": 0.5583,
}


def _last_row(path: Path) -> dict[str, str]:
    """读取 CSV 最后一行，用作 After R3 指标。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"结果为空：{path}")
    return rows[-1]


def _metric(row: dict[str, str], *keys: str) -> float:
    """兼容不同阶段 CSV 的列名差异，找不到时返回 NaN。"""
    for key in keys:
        if key in row and str(row[key]).strip() != "":
            try:
                return float(row[key])
            except ValueError:
                return float("nan")
    return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 ADS-B 高置信伪标签注册 seed7 结果。")
    parser.add_argument("--root", required=True, help="Stage 17 结果根目录。")
    parser.add_argument("--output", required=True, help="Markdown 报告输出路径。")
    args = parser.parse_args()

    root = Path(args.root)
    candidates = sorted(root.glob("top*/incremental_results.csv"))
    if not candidates:
        raise FileNotFoundError(f"未找到候选结果：{root}/top*/incremental_results.csv")

    rows = []
    for path in candidates:
        row = _last_row(path)
        name = path.parent.name
        overall = _metric(row, "Overall Acc")
        old = _metric(row, "Old Acc")
        new = _metric(row, "New Acc")
        forgetting = _metric(row, "Forgetting", "Forgetting Rate")
        macro_f1 = _metric(row, "Macro F1")
        rows.append(
            {
                "name": name,
                "overall": overall,
                "old": old,
                "new": new,
                "forgetting": forgetting,
                "macro_f1": macro_f1,
                "delta_overall": overall - BASELINE_R3["Overall Acc"],
                "delta_old": old - BASELINE_R3["Old Acc"],
                "delta_new": new - BASELINE_R3["New Acc"],
            }
        )

    rows.sort(key=lambda item: item["overall"], reverse=True)
    best = rows[0]
    passed = (
        best["overall"] > BASELINE_R3["Overall Acc"]
        and best["old"] >= BASELINE_R3["Old Acc"] - 0.01
        and best["new"] >= BASELINE_R3["New Acc"] - 0.03
    )

    lines = [
        "# Stage 17 ADS-B 高置信伪标签注册 seed7 报告",
        "",
        "组合：cross_day 训练期表征适配 + GPCC discovery + 高置信伪标签注册 + RADCIL。",
        "",
        f"seed7 对照：Overall={BASELINE_R3['Overall Acc']:.4f}, Old={BASELINE_R3['Old Acc']:.4f}, New={BASELINE_R3['New Acc']:.4f}。",
        "",
        "| Candidate | R3 Overall | Delta Overall | R3 Old | Delta Old | R3 New | Delta New | Forgetting | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {overall:.4f} | {delta_overall:+.4f} | {old:.4f} | {delta_old:+.4f} | "
            "{new:.4f} | {delta_new:+.4f} | {forgetting:.4f} | {macro_f1:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "判定："
            + (
                f"{best['name']} 通过 seed7 门槛，可进入 ADS-B 三种子确认。"
                if passed
                else f"{best['name']} 未通过 seed7 门槛，高置信伪标签注册先归档为负消融。"
            ),
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
