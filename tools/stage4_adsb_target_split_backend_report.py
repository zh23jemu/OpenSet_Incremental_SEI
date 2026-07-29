"""汇总 ADS-B target split 后端遗忘对照结果。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.stage3_wisig_cil_baseline_report import (  # noqa: E402
    aggregate,
    collect_seed,
    find_method,
    parse_seeds,
    render_metric_table,
    table_rows,
)
from tools.stage4_adsb_cil_baseline_report import collect_frontend  # noqa: E402


def fmt_delta(value: float) -> str:
    """统一输出带符号的四位小数差值。"""

    return f"{value:+.4f}"


def load_baseline_summary(path: Path | None) -> list[dict[str, Any]]:
    """读取旧 ADS-B baseline 聚合表；缺省时返回空列表。"""

    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"缺少旧 baseline 汇总 JSON：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("summary", []))


def metric(row: dict[str, Any] | None, key: str) -> float | None:
    """从聚合行读取均值指标，缺失时返回 None。"""

    if row is None:
        return None
    value = row.get(f"{key}_mean")
    return None if value is None else float(value)


def build_report(
    job_id: str,
    seeds: tuple[int, ...],
    output_prefix: str,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    frontend: list[dict[str, Any]],
    baseline_summary: list[dict[str, Any]],
) -> str:
    """生成 Markdown 报告，重点回答 target split 后遗忘风险是否收敛。"""

    main = find_method(summary, "main_method", "MV-ACC-CIL")
    shared = table_rows(summary, "shared_discovery_backend")
    end_to_end = table_rows(summary, "end_to_end_deep_hdbscan")
    doi = find_method(summary, "shared_discovery_backend", "DOI-style")
    old_main = find_method(baseline_summary, "main_method", "MV-ACC-CIL")
    old_doi = find_method(baseline_summary, "shared_discovery_backend", "DOI-style")

    lines = [
        "# 阶段 4 ADS-B Target Split 后端遗忘对照",
        "",
        f"- 生成时间 UTC：{datetime.now(timezone.utc).isoformat()}",
        f"- Slurm Job：`{job_id}`；seeds：`{', '.join(str(seed) for seed in seeds)}`。",
        f"- 输出前缀：`{output_prefix}`",
        "- 协议：ADS-B strict 90+10x3；长序列 backbone；固定 target split `clusters=10/max_added=4/silhouette=0.26`；打开同协议 CIL baselines。",
        "- 目的：在欠聚类已被 target split 部分缓解后，继续判断 ADS-B Forgetting 高于 DOI-style 的风险是否来自 RADCIL 后端权衡。",
        "",
        "## Target Split 主方法",
        "",
        *render_metric_table([main] if main is not None else []),
        "",
        "## Target Split 共享发现后端 Baseline",
        "",
        *render_metric_table(shared),
        "",
        "## Target Split Deep-HDBSCAN 端到端 Baseline",
        "",
        *render_metric_table(end_to_end),
        "",
        "## 与旧发现条件对照",
        "",
    ]

    if main is not None and old_main is not None:
        lines.append(
            "- MV-ACC-CIL target split 相对旧发现条件："
            f"Overall {fmt_delta(float(metric(main, 'overall') or 0.0) - float(metric(old_main, 'overall') or 0.0))}，"
            f"New {fmt_delta(float(metric(main, 'new') or 0.0) - float(metric(old_main, 'new') or 0.0))}，"
            f"Forgetting {fmt_delta(float(metric(main, 'forgetting') or 0.0) - float(metric(old_main, 'forgetting') or 0.0))}。"
        )
    if doi is not None and old_doi is not None:
        lines.append(
            "- DOI-style target split 相对旧发现条件："
            f"Overall {fmt_delta(float(metric(doi, 'overall') or 0.0) - float(metric(old_doi, 'overall') or 0.0))}，"
            f"New {fmt_delta(float(metric(doi, 'new') or 0.0) - float(metric(old_doi, 'new') or 0.0))}，"
            f"Forgetting {fmt_delta(float(metric(doi, 'forgetting') or 0.0) - float(metric(old_doi, 'forgetting') or 0.0))}。"
        )
    if main is not None and doi is not None:
        lines.append(
            "- Target split 同发现条件下 MV-ACC-CIL 相对 DOI-style："
            f"Overall {fmt_delta(float(metric(main, 'overall') or 0.0) - float(metric(doi, 'overall') or 0.0))}，"
            f"New {fmt_delta(float(metric(main, 'new') or 0.0) - float(metric(doi, 'new') or 0.0))}，"
            f"Forgetting {fmt_delta(float(metric(main, 'forgetting') or 0.0) - float(metric(doi, 'forgetting') or 0.0))}。"
        )

    lines.extend([
        "",
        "## 发现前端审计",
        "",
        "| Seed | 前端 | 轮次 | 簇数 | NMI | ARI | Hungarian |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in sorted(frontend, key=lambda item: (item["seed"], item["method"], item["round"])):
        lines.append(
            f"| {row['seed']} | {row['method']} | {row['round']} | {row['clusters']} | "
            f"{row['nmi']:.4f} | {row['ari']:.4f} | {row['hungarian']:.4f} |"
        )

    lines.extend([
        "",
        "## 判定边界",
        "",
        "- 本报告只聚合已完成实验结果；真实标签指标只用于事后审计，不参与 target split 或 baseline 选择。",
        "- 若 DOI-style 继续保持低 Forgetting 但 New/Overall 明显偏低，后续应把 ADS-B 风险写成旧/新后端权衡，而不是继续搜索 target split 小参数。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 ADS-B target split 后端遗忘对照。")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--seeds", default="7,13,31")
    parser.add_argument("--output-prefix", default="results/stage4/adsb_target_split_backend_baselines")
    parser.add_argument("--baseline-summary-json", default="results/stage4/stage4_adsb_cil_baselines_summary_44517860.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    rows: list[dict[str, Any]] = []
    frontend: list[dict[str, Any]] = []
    for seed in seeds:
        save_dir = Path(f"{args.output_prefix}_seed{seed}_{args.job_id}")
        rows.extend(collect_seed(seed, save_dir))
        frontend.extend(collect_frontend(seed, save_dir))

    summary = aggregate(rows)
    baseline_summary = load_baseline_summary(Path(args.baseline_summary_json) if args.baseline_summary_json else None)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_report(args.job_id, seeds, args.output_prefix, rows, summary, frontend, baseline_summary),
        encoding="utf-8",
    )

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "job_id": args.job_id,
                "seeds": list(seeds),
                "output_prefix": args.output_prefix,
                "baseline_summary_json": args.baseline_summary_json,
                "per_seed_r3": rows,
                "summary": summary,
                "frontend": frontend,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"ADS-B target split 后端遗忘对照报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
