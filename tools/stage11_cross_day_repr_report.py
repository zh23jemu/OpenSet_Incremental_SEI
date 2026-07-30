"""汇总 Stage 11 跨天表征适配 discovery-only 结果。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    """读取单个方法的三轮聚类结果；缺失目录时返回空列表。"""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _mean(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return sum(valid) / len(valid) if valid else float("nan")


def _summarize(dataset: str, method: str, rows: list[dict[str, str]]) -> dict[str, object]:
    """计算报告中用于门槛判断的均值、R3 和簇数审计字段。"""
    hungarian = [_number(row, "Hungarian Acc") for row in rows]
    purity = [_number(row, "Purity") for row in rows]
    clusters = [_number(row, "Final Cluster Count") for row in rows]
    return {
        "dataset": dataset,
        "method": method,
        "rounds": len(rows),
        "mean_hungarian": _mean(hungarian),
        "mean_purity": _mean(purity),
        "r3_hungarian": hungarian[-1] if hungarian else float("nan"),
        "r3_purity": purity[-1] if purity else float("nan"),
        "cluster_counts": clusters,
    }


def _fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 Stage 11 跨天表征适配结果。")
    parser.add_argument("--root", default="results/stage11")
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    summaries = []
    for dataset in ("lora25", "adsb"):
        for method in ("none", "cross_day"):
            directory = root / f"{dataset}_{method}_seed7_{args.job_id}"
            summaries.append(
                _summarize(dataset, method, _read_rows(directory / "clustering_results.csv"))
            )

    lines = [
        "# Stage 11 跨天表征适配 discovery-only seed7 报告",
        "",
        "本报告只读取 discovery-only 的 clustering_results.csv；held-out eval 不参与候选选择。",
        "",
        "| 数据集 | 方法 | 轮数 | Mean Hungarian | Mean Purity | R3 Hungarian | R3 Purity | 簇数 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['dataset']} | {item['method']} | {item['rounds']} | "
            f"{_fmt(item['mean_hungarian'])} | {_fmt(item['mean_purity'])} | "
            f"{_fmt(item['r3_hungarian'])} | {_fmt(item['r3_purity'])} | "
            f"{item['cluster_counts']} |"
        )
    lines.extend(
        [
            "",
            "判定：LoRa 重点看三轮 mean Hungarian/Purity，ADS-B 重点看 R3；"
            "不过门槛不进入完整 CIL，也不扩展其他 seed。",
        ]
    )

    output = Path(
        args.output
        or root / f"STAGE11_CROSS_DAY_REPR_SEED7_REPORT_{args.job_id}.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
