"""生成阶段 6 GPCC seed7 发现前端对比报告。

本报告只读取 discovery-only 运行产生的 ``clustering_results.csv``，
用于比较旧 MV-ACC/HDBSCAN 前端和新 GPCC 前端的聚类质量。报告不读取
held-out evaluation 预测结果，也不依赖任何增量后端输出。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "Final Cluster Count",
    "True New Classes",
    "Cluster Count Error",
    "Noise Points",
    "Assignment Coverage",
    "Purity",
    "NMI",
    "ARI",
    "Hungarian Acc",
    "GPCC Uses HDBSCAN",
    "GPCC Cluster Size CV",
    "GPCC Label-Free Score",
]


def _to_float(value: Any, default: float = float("nan")) -> float:
    """把 CSV 中可能出现的字符串/空值安全转换成浮点数。"""

    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any) -> str:
    """统一指标显示格式，避免 Markdown 表格里出现过长小数。"""

    number = _to_float(value)
    if np.isnan(number):
        return str(value)
    return f"{number:.4f}"


def load_frontend_rows(label: str, save_dir: Path) -> list[dict[str, Any]]:
    """读取一个前端的 discovery-only 输出，并补充 variant/save_dir 字段。"""

    csv_path = save_dir / "clustering_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"缺少聚类结果：{csv_path}")
    df = pd.read_csv(csv_path)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        item = {
            "Variant": label,
            "Save Dir": str(save_dir),
            "Round": str(row.get("Round", "")),
        }
        for col in METRIC_COLUMNS:
            if col in row:
                item[col] = row[col]
        rows.append(item)
    return rows


def summarize_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合单个前端的三轮聚类质量，保留通过门槛需要的关键指标。"""

    if not rows:
        raise ValueError("rows 不能为空")
    cluster_errors = [_to_float(r.get("Cluster Count Error")) for r in rows]
    ari = [_to_float(r.get("ARI")) for r in rows]
    hungarian = [_to_float(r.get("Hungarian Acc")) for r in rows]
    noise = [_to_float(r.get("Noise Points"), 0.0) for r in rows]
    # GPCC 固定为全样本分配簇，旧 CSV 若没有显式 coverage 字段，按 1.0
    # 处理；MV-ACC 旧前端缺少 GPCC 标记时，按历史实现视为使用 HDBSCAN。
    coverage = [_to_float(r.get("Assignment Coverage"), 1.0) for r in rows]
    uses_hdbscan_values = [str(r.get("GPCC Uses HDBSCAN", "")).lower() for r in rows]
    default_uses_hdbscan = rows[0]["Variant"] != "gpcc"
    return {
        "variant": rows[0]["Variant"],
        "rounds": len(rows),
        "all_target_cluster_count": bool(all(v == 0 for v in cluster_errors)),
        "total_cluster_count_error": float(np.nansum(cluster_errors)),
        "mean_ari": float(np.nanmean(ari)),
        "mean_hungarian_acc": float(np.nanmean(hungarian)),
        "total_noise_points": float(np.nansum(noise)),
        "mean_assignment_coverage": float(np.nanmean(coverage)),
        "uses_hdbscan": any(v == "true" for v in uses_hdbscan_values) or (
            default_uses_hdbscan and all(v in {"", "nan", "none"} for v in uses_hdbscan_values)
        ),
    }


def compare_gpcc_to_baseline(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """按预注册门槛比较 GPCC 和旧前端。"""

    if "gpcc" not in summary:
        return {"has_gpcc": False, "passed_gate": False, "reason": "missing_gpcc"}
    gpcc = summary["gpcc"]
    baseline = summary.get("mvacc")
    result = {
        "has_gpcc": True,
        "gpcc_cluster_count_gate": bool(gpcc["all_target_cluster_count"]),
        "gpcc_no_noise_gate": bool(gpcc["total_noise_points"] == 0),
        "gpcc_uses_hdbscan": bool(gpcc["uses_hdbscan"]),
    }
    if baseline is None:
        result.update({"passed_gate": False, "reason": "missing_mvacc_baseline"})
        return result
    ari_delta = float(gpcc["mean_ari"] - baseline["mean_ari"])
    hungarian_delta = float(gpcc["mean_hungarian_acc"] - baseline["mean_hungarian_acc"])
    result.update({
        "ari_delta_vs_mvacc": ari_delta,
        "hungarian_delta_vs_mvacc": hungarian_delta,
        "passed_gate": bool(
            gpcc["all_target_cluster_count"]
            and gpcc["total_noise_points"] == 0
            and not gpcc["uses_hdbscan"]
            and (ari_delta > 0.01 or hungarian_delta > 0.01)
            and ari_delta > -0.01
            and hungarian_delta > -0.01
        ),
    })
    return result


def build_markdown(dataset: str, job_id: str, rows: list[dict[str, Any]], summary: dict[str, Any], gate: dict[str, Any]) -> str:
    """构造面向客户/项目汇报的简洁 Markdown 报告。"""

    lines = [
        f"# 阶段 6 {dataset} GPCC seed7 聚类前端报告",
        "",
        f"- 生成时间 UTC：{datetime.now(timezone.utc).isoformat()}",
        f"- Slurm Job：{job_id}",
        "- 运行模式：discovery-only，只验证聚类，不跑增量后端。",
        "- 协议边界：聚类前端只使用已知 train/validation 与当前 discovery；held-out eval 不参与特征标准化、聚类或选择。",
        "",
        "## 结论",
        "",
    ]
    if gate.get("passed_gate"):
        lines.append("- GPCC 通过 seed7 聚类门槛，可进入完整增量 seed7 验证。")
    else:
        lines.append("- GPCC 暂未通过 seed7 聚类门槛，先不要扩三种子。")
    if "ari_delta_vs_mvacc" in gate:
        lines.append(f"- 相对 MV-ACC：mean ARI {gate['ari_delta_vs_mvacc']:+.4f}，mean Hungarian Acc {gate['hungarian_delta_vs_mvacc']:+.4f}。")
    lines.extend(["", "## 聚合指标", ""])
    lines.append("| Variant | Target Count OK | Noise | Mean ARI | Mean Hungarian | Mean Coverage | Uses HDBSCAN |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for variant, item in summary.items():
        lines.append(
            "| {variant} | {target} | {noise} | {ari} | {hungarian} | {coverage} | {hdbscan} |".format(
                variant=variant,
                target=str(item["all_target_cluster_count"]),
                noise=_fmt(item["total_noise_points"]),
                ari=_fmt(item["mean_ari"]),
                hungarian=_fmt(item["mean_hungarian_acc"]),
                coverage=_fmt(item["mean_assignment_coverage"]),
                hdbscan=str(item["uses_hdbscan"]),
            )
        )
    lines.extend(["", "## 分轮指标", ""])
    lines.append("| Variant | Round | Final K | Target K | Error | Noise | ARI | Hungarian | Coverage |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| {variant} | {round} | {final_k} | {target_k} | {error} | {noise} | {ari} | {hungarian} | {coverage} |".format(
                variant=row["Variant"],
                round=row["Round"],
                final_k=_fmt(row.get("Final Cluster Count")),
                target_k=_fmt(row.get("True New Classes")),
                error=_fmt(row.get("Cluster Count Error")),
                noise=_fmt(row.get("Noise Points", 0.0)),
                ari=_fmt(row.get("ARI")),
                hungarian=_fmt(row.get("Hungarian Acc")),
                coverage=_fmt(row.get("Assignment Coverage", 1.0)),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总阶段 6 GPCC discovery-only seed7 对比。")
    parser.add_argument("--dataset", required=True, choices=["adsb", "lora"])
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--mvacc-save-dir", required=True)
    parser.add_argument("--gpcc-save-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    rows = []
    rows.extend(load_frontend_rows("mvacc", Path(args.mvacc_save_dir)))
    rows.extend(load_frontend_rows("gpcc", Path(args.gpcc_save_dir)))
    grouped = {
        label: summarize_variant([row for row in rows if row["Variant"] == label])
        for label in ["mvacc", "gpcc"]
    }
    gate = compare_gpcc_to_baseline(grouped)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(args.dataset, args.job_id, rows, grouped, gate), encoding="utf-8")

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "job_id": args.job_id,
                "rows": rows,
                "summary": grouped,
                "gate": gate,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"GPCC discovery report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
