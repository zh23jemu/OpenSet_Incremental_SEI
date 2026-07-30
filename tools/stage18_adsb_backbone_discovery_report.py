"""汇总 Stage 18 ADS-B discovery backbone 重训 seed7 结果。

Stage 18 只评估发现前端质量，不跑增量训练。候选 checkpoint 只用 Day1 已知类
训练/验证生成，随后在 R1-R3 discovery 样本上用 GPCC 固定输出协议簇数。
报告只读取 clustering_results.csv 小结果，避免把后端和 held-out eval 因素混入。
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


STATIC_GPCC_R3_HUNGARIAN = 0.6270
STATIC_GPCC_R3_PURITY = 0.7337
CROSS_DAY_R3_HUNGARIAN = 0.6074
CROSS_DAY_R3_PURITY = 0.7283


def _read_rows(path: Path) -> list[dict[str, str]]:
    """读取一个候选的三轮 discovery-only 聚类结果。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"结果为空：{path}")
    return rows


def _number(row: dict[str, str], key: str) -> float:
    """安全读取浮点指标。"""
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _mean(values: list[float]) -> float:
    """忽略 NaN 后求均值。"""
    valid = [value for value in values if not math.isnan(value)]
    return sum(valid) / len(valid) if valid else float("nan")


def _fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 ADS-B discovery backbone 重训结果。")
    parser.add_argument("--root", required=True, help="Stage 18 结果根目录。")
    parser.add_argument("--output", required=True, help="Markdown 报告输出路径。")
    args = parser.parse_args()

    root = Path(args.root)
    candidates = sorted(root.glob("*/clustering_results.csv"))
    if not candidates:
        raise FileNotFoundError(f"未找到候选结果：{root}/*/clustering_results.csv")

    summaries = []
    for path in candidates:
        rows = _read_rows(path)
        hungarian = [_number(row, "Hungarian Acc") for row in rows]
        purity = [_number(row, "Purity") for row in rows]
        clusters = [_number(row, "Final Cluster Count") for row in rows]
        summaries.append(
            {
                "name": path.parent.name,
                "rounds": len(rows),
                "mean_hungarian": _mean(hungarian),
                "mean_purity": _mean(purity),
                "r3_hungarian": hungarian[-1],
                "r3_purity": purity[-1],
                "cluster_counts": clusters,
                "delta_static_h": hungarian[-1] - STATIC_GPCC_R3_HUNGARIAN,
                "delta_cross_h": hungarian[-1] - CROSS_DAY_R3_HUNGARIAN,
            }
        )

    summaries.sort(key=lambda item: item["r3_hungarian"], reverse=True)
    best = summaries[0]
    passed = (
        best["r3_hungarian"] >= STATIC_GPCC_R3_HUNGARIAN + 0.015
        and best["r3_purity"] >= STATIC_GPCC_R3_PURITY - 0.010
        and all(abs(value - 10.0) < 1e-6 for value in best["cluster_counts"])
    )

    lines = [
        "# Stage 18 ADS-B Discovery Backbone 重训 seed7 报告",
        "",
        "本报告只评估 GPCC discovery-only 聚类质量；held-out eval 和 RADCIL 后端不参与候选选择。",
        "",
        (
            "对照：static GPCC R3 Hungarian/Purity="
            f"{STATIC_GPCC_R3_HUNGARIAN:.4f}/{STATIC_GPCC_R3_PURITY:.4f}；"
            "cross_day GPCC R3 Hungarian/Purity="
            f"{CROSS_DAY_R3_HUNGARIAN:.4f}/{CROSS_DAY_R3_PURITY:.4f}。"
        ),
        "",
        "| Candidate | Rounds | Mean Hungarian | Mean Purity | R3 Hungarian | Delta Static R3 | Delta Cross-Day R3 | R3 Purity | 簇数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['name']} | {item['rounds']} | {_fmt(item['mean_hungarian'])} | "
            f"{_fmt(item['mean_purity'])} | {_fmt(item['r3_hungarian'])} | "
            f"{item['delta_static_h']:+.4f} | {item['delta_cross_h']:+.4f} | "
            f"{_fmt(item['r3_purity'])} | {item['cluster_counts']} |"
        )
    lines.extend(
        [
            "",
            "判定："
            + (
                f"{best['name']} 通过 discovery 门槛，可进入 seed7 完整 CIL。"
                if passed
                else f"{best['name']} 未通过 discovery 门槛，不进入完整 CIL。"
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
