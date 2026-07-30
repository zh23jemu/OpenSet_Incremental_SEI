"""汇总 Stage 16 ADS-B 旧/新双分支 seed7 矩阵结果。

该报告器只读取每个候选目录下的 CSV 小结果，不读取模型权重、replay memory
或 held-out eval 以外的训练中间产物。候选是否继续扩三种子，按 R3 Overall
是否超过 Stage 12/13 seed7 cross_day + GPCC 对照，并检查 Old/New 是否塌缩。
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
    """读取 incremental_results.csv 最后一行，也就是 After R3 指标。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"结果为空：{path}")
    return rows[-1]


def _to_float(row: dict[str, str], key: str) -> float:
    """把 CSV 字符串安全转为浮点数，缺失时返回 NaN。"""
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def _metric(row: dict[str, str], *keys: str) -> float:
    """按候选字段名读取指标，兼容不同阶段 CSV 的列名微小差异。"""
    for key in keys:
        if key in row and str(row[key]).strip() != "":
            return _to_float(row, key)
    return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 ADS-B 旧/新双分支 seed7 结果。")
    parser.add_argument("--root", required=True, help="Stage 16 结果根目录。")
    parser.add_argument("--output", required=True, help="Markdown 报告输出路径。")
    args = parser.parse_args()

    root = Path(args.root)
    candidates = sorted(root.glob("branch_w*/incremental_results.csv"))
    if not candidates:
        raise FileNotFoundError(f"未找到候选结果：{root}/branch_w*/incremental_results.csv")

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
        "# Stage 16 ADS-B 旧/新双分支 seed7 报告",
        "",
        "组合：cross_day 训练期表征适配 + GPCC discovery + RADCIL + 旧/新分支判别损失。",
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
                else f"{best['name']} 未通过 seed7 门槛，旧/新双分支先归档为负消融。"
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
