"""汇总 ADS-B strict CIL baseline 三种子结果与前端审计。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 保证从项目根目录或任意工作目录执行时都能复用阶段 3 的聚合函数。
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.stage3_wisig_cil_baseline_report import (
    aggregate,
    collect_seed,
    find_method,
    render_metric_table,
    table_rows,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def collect_frontend(seed: int, save_dir: Path) -> list[dict[str, Any]]:
    """收集主 MV-ACC 和中性 Deep-HDBSCAN 的每轮发现质量。"""

    rows = []
    sources = (
        ("MV-ACC", save_dir / "clustering_results.csv"),
        ("Deep-HDBSCAN", save_dir / "deep_hdbscan_clustering_results.csv"),
    )
    for method, path in sources:
        for row in read_csv(path):
            rows.append({
                "seed": seed,
                "method": method,
                "round": row["Round"],
                "clusters": int(float(row["Final Cluster Count"])),
                "nmi": float(row["NMI"]),
                "ari": float(row["ARI"]),
                "hungarian": float(row["Hungarian Acc"]),
            })
    return rows


def build_report(job_id: str, seeds: list[int], summary: list[dict[str, Any]], frontend: list[dict[str, Any]]) -> str:
    main = find_method(summary, "main_method", "MV-ACC-CIL")
    shared = table_rows(summary, "shared_discovery_backend")
    end_to_end = table_rows(summary, "end_to_end_deep_hdbscan")
    lines = [
        "# 阶段 4 ADS-B Strict Baseline 与前端消融表",
        "",
        f"- 生成时间 UTC：{datetime.now(timezone.utc).isoformat()}",
        f"- Slurm Job：`{job_id}`；seeds：`{', '.join(map(str, seeds))}`。",
        "- 协议：ADS-B strict 90+10x3；长序列 backbone；正式 ratio 0.03；old:new=2.0；replay weight=3.0。",
        "",
        "## 主方法",
        "",
        *render_metric_table([main] if main is not None else []),
        "",
        "## 共享 MV-ACC 发现的后端 Baseline",
        "",
        *render_metric_table(shared),
        "",
        "## Deep-HDBSCAN 端到端 Baseline",
        "",
        *render_metric_table(end_to_end),
        "",
        "## 发现前端逐轮审计",
        "",
        "| Seed | 前端 | 轮次 | 簇数 | NMI | ARI | Hungarian |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(frontend, key=lambda item: (item["seed"], item["method"], item["round"])):
        lines.append(
            f"| {row['seed']} | {row['method']} | {row['round']} | {row['clusters']} | "
            f"{row['nmi']:.4f} | {row['ari']:.4f} | {row['hungarian']:.4f} |"
        )
    lines.extend([
        "",
        "## 判定边界",
        "",
        "前端 NMI/ARI/Hungarian 与增量准确率均为实验完成后的离线审计指标，",
        "不参与 discovery 密度选择、训练或 checkpoint 选择。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 ADS-B strict baseline 三种子结果。")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--seeds", default="7,13,31")
    parser.add_argument("--output-prefix", default="results/stage4/adsb_cil_baselines")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    rows = []
    frontend = []
    for seed in seeds:
        save_dir = Path(f"{args.output_prefix}_seed{seed}_{args.job_id}")
        rows.extend(collect_seed(seed, save_dir))
        frontend.extend(collect_frontend(seed, save_dir))
    summary = aggregate(rows)
    Path(args.output).write_text(build_report(args.job_id, seeds, summary, frontend), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {"job_id": args.job_id, "seeds": seeds, "per_seed_r3": rows, "summary": summary, "frontend": frontend},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"ADS-B strict baseline 三种子报告：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
