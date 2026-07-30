"""汇总 Stage 10 discovery 特征适配器结果。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _summarize(dataset: str, adapter: str, rows: list[dict[str, str]]) -> dict[str, object]:
    hungarian = [_number(row, "Hungarian Acc") for row in rows]
    purity = [_number(row, "Purity") for row in rows]
    clusters = [_number(row, "Final Cluster Count") for row in rows]
    return {
        "dataset": dataset,
        "adapter": adapter,
        "rounds": len(rows),
        "mean_hungarian": sum(hungarian) / len(hungarian) if hungarian else float("nan"),
        "mean_purity": sum(purity) / len(purity) if purity else float("nan"),
        "r3_hungarian": hungarian[-1] if hungarian else float("nan"),
        "r3_purity": purity[-1] if purity else float("nan"),
        "cluster_counts": clusters,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 Stage 10 discovery 特征适配器结果。")
    parser.add_argument("--root", default="results/stage10")
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    summaries = []
    for dataset in ("lora25", "adsb"):
        for adapter in ("none", "mn_smooth", "proto_repulse"):
            directory = root / f"{dataset}_{adapter}_seed7_{args.job_id}"
            summaries.append(_summarize(dataset, adapter, _read_rows(directory / "clustering_results.csv")))

    output = Path(args.output or root / f"STAGE10_DISCOVERY_FEATURE_ADAPTER_SEED7_REPORT_{args.job_id}.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 10 discovery 特征适配器 seed7 报告",
        "",
        "本报告只读取 discovery-only 的 clustering_results.csv；held-out eval 不参与候选选择。",
        "",
        "| 数据集 | 适配器 | 轮数 | Mean Hungarian | Mean Purity | R3 Hungarian | R3 Purity | 簇数 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        fmt = lambda value: "nan" if value != value else f"{value:.4f}"
        lines.append(
            f"| {item['dataset']} | {item['adapter']} | {item['rounds']} | "
            f"{fmt(item['mean_hungarian'])} | {fmt(item['mean_purity'])} | "
            f"{fmt(item['r3_hungarian'])} | {fmt(item['r3_purity'])} | "
            f"{item['cluster_counts']} |"
        )
    lines.extend([
        "",
        "判定：LoRa 先看三轮 mean Hungarian/Purity，ADS-B 重点看 R3；不过门槛不进入完整 CIL。",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
