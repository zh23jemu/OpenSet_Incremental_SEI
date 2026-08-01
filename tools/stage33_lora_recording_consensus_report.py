"""汇总 Stage 33 LoRa recording-consensus discovery-only 结果。

报告只读取每个变体的 ``clustering_results.csv``。聚类器训练和选择阶段
不使用 held-out IQ_8-10 真值；这里的 ARI/Hungarian 只用于实验结束后的
离线质量比较，不会回流到训练流程。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


VARIANTS = ("gpcc", "consensus_t0p65", "consensus_t0p75", "consensus_t0p85")


def _float(value: Any, default: float = float("nan")) -> float:
    """把 CSV 字段安全转换为浮点数。"""

    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: float) -> str:
    """统一报告格式。"""

    return "nan" if math.isnan(value) else f"{value:.4f}"


def _load_variant(root: Path, variant: str, job_id: str) -> list[dict[str, Any]]:
    """读取单个变体的聚类表，并补充变体名。"""

    path = root / f"lora_recording_consensus_{variant}_seed7_{job_id}" / "clustering_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少 Stage 33 聚类结果：{path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Stage 33 聚类结果为空：{path}")
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "variant": variant,
                "round": str(row.get("Round", "")),
                "final_k": _float(row.get("Final Cluster Count")),
                "target_k": _float(row.get("True New Classes")),
                "cluster_error": _float(row.get("Cluster Count Error")),
                "noise": _float(row.get("Noise Points"), 0.0),
                "coverage": _float(row.get("Assignment Coverage"), 1.0),
                "ari": _float(row.get("ARI")),
                "hungarian": _float(row.get("Hungarian Acc")),
                "purity": _float(row.get("Purity")),
            }
        )
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合三轮指标并检查固定簇数/无噪声约束。"""

    return {
        "rounds": len(rows),
        "target_count_ok": all(
            int(row["final_k"]) == int(row["target_k"]) for row in rows
        ),
        "zero_noise": all(row["noise"] == 0.0 for row in rows),
        "mean_ari": float(sum(row["ari"] for row in rows) / len(rows)),
        "mean_hungarian": float(sum(row["hungarian"] for row in rows) / len(rows)),
        "mean_purity": float(sum(row["purity"] for row in rows) / len(rows)),
        "mean_coverage": float(sum(row["coverage"] for row in rows) / len(rows)),
        "total_cluster_error": float(sum(row["cluster_error"] for row in rows)),
    }


def _gate(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """按预注册规则判断是否值得进入完整 CIL。"""

    baseline = summary["gpcc"]
    candidates = {}
    for name in VARIANTS[1:]:
        current = summary[name]
        ari_delta = current["mean_ari"] - baseline["mean_ari"]
        hungarian_delta = current["mean_hungarian"] - baseline["mean_hungarian"]
        passed = bool(
            current["target_count_ok"]
            and current["zero_noise"]
            and (ari_delta > 0.01 or hungarian_delta > 0.01)
            and ari_delta > -0.01
            and hungarian_delta > -0.01
        )
        candidates[name] = {
            "ari_delta_vs_gpcc": ari_delta,
            "hungarian_delta_vs_gpcc": hungarian_delta,
            "passed": passed,
        }
    return {"baseline": "gpcc", "candidates": candidates}


def main() -> int:
    """读取四个变体并生成 Markdown/JSON 报告。"""

    parser = argparse.ArgumentParser(description="汇总 Stage 33 LoRa 聚类结果")
    parser.add_argument("--root", default="results/stage33")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        variant_rows = _load_variant(root, variant, str(args.job_id))
        rows.extend(variant_rows)
        summary[variant] = _summarize(variant_rows)
    gate = _gate(summary)
    passed = [name for name, item in gate["candidates"].items() if item["passed"]]

    lines = [
        "# Stage 33 LoRa Recording-Consensus discovery-only 报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 固定条件：LoRa Chirp、cross-day 表征适配、LoRa SSL、strict split、discovery-only。",
        "- 目的：验证 symbol 级 GPCC 后的选择性 recording 共识，避免 Stage 24 的整段平均压掉 symbol 细节。",
        "- 边界：held-out IQ_8-10 不参与标准化、聚类或参数选择；真值只用于离线指标。",
        "",
        "## 聚合结果",
        "",
        "| Variant | Target K | Zero Noise | Mean ARI | Mean Hungarian | Mean Purity | Mean Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANTS:
        item = summary[name]
        lines.append(
            f"| {name} | {item['target_count_ok']} | {item['zero_noise']} | "
            f"{_fmt(item['mean_ari'])} | {_fmt(item['mean_hungarian'])} | "
            f"{_fmt(item['mean_purity'])} | {_fmt(item['mean_coverage'])} |"
        )
    lines.extend(["", "## 相对普通 GPCC 的门槛", ""])
    for name, item in gate["candidates"].items():
        lines.append(
            f"- `{name}`：ARI `{item['ari_delta_vs_gpcc']:+.4f}`，"
            f"Hungarian `{item['hungarian_delta_vs_gpcc']:+.4f}`，"
            f"通过=`{item['passed']}`。"
        )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            (
                f"- 通过变体：{', '.join(passed)}；可以只对通过者提交完整 seed7 CIL。"
                if passed
                else "- 没有变体通过 discovery 门槛；不进入 CIL，转向联合 discovery-CIL self-training。"
            ),
            "",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "stage33_lora_recording_consensus_summary_v1",
                "job_id": str(args.job_id),
                "summary": summary,
                "gate": gate,
                "passed_variants": passed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Stage 33 报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
