"""Stage 9 前端瓶颈诊断报告。

这个脚本不启动训练，也不读取 held-out 样本之外的新数据。它只汇总
已经完成实验中的 CSV / JSON 小结果，用于判断 ADS-B / LoRa 当前低分
主要卡在发现前端、伪标签纯度，还是增量后端。诊断结果只作为下一轮
方案选择依据，不用于调参或回写任何模型。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: str | Path) -> pd.DataFrame:
    """读取项目内 CSV，并在缺文件时给出清晰错误。"""
    full_path = ROOT / path
    if not full_path.exists():
        raise FileNotFoundError(f"缺少诊断输入：{full_path}")
    return pd.read_csv(full_path)


def _round_rows(clustering_path: str | Path, incremental_path: str | Path, dataset: str, variant: str) -> list[dict[str, Any]]:
    """把聚类表和增量表按 R1/R2/R3 合并成诊断行。"""
    clustering = _read_csv(clustering_path)
    incremental = _read_csv(incremental_path)
    rows: list[dict[str, Any]] = []
    for round_name in ("R1", "R2", "R3"):
        cluster_row = clustering[clustering["Round"].astype(str) == round_name]
        stage_row = incremental[incremental["Stage"].astype(str) == f"After {round_name}"]
        if cluster_row.empty or stage_row.empty:
            continue
        c = cluster_row.iloc[0]
        s = stage_row.iloc[0]
        hungarian = float(c.get("Hungarian Acc", float("nan")))
        purity = float(c.get("Purity", float("nan")))
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "round": round_name,
                "cluster_count": int(c.get("Final Cluster Count", c.get("Cluster Count", -1))),
                "cluster_count_error": int(c.get("Cluster Count Error", -1)),
                "purity": purity,
                "ari": float(c.get("ARI", float("nan"))),
                "hungarian_acc": hungarian,
                # Hungarian Acc 可理解为聚类到真实类别的一对一后验上界，
                # 1-Hungarian Acc 是伪标签噪声的保守下界。
                "pseudo_noise_floor": 1.0 - hungarian,
                "overall": float(s.get("Overall Acc", float("nan"))),
                "old": float(s.get("Old Acc", float("nan"))),
                "new": float(s.get("New Acc", float("nan"))),
                "forgetting": float(s.get("Forgetting Rate", float("nan"))),
            }
        )
    return rows


def collect_rows() -> list[dict[str, Any]]:
    """汇总当前最能代表 ADS-B / LoRa 风险的已有实验。"""
    rows: list[dict[str, Any]] = []
    rows.extend(
        _round_rows(
            "results/stage4/adsb_target_split_backend_baselines_seed7_44780602/clustering_results.csv",
            "results/stage4/adsb_target_split_backend_baselines_seed7_44780602/incremental_results.csv",
            "ADS-B",
            "target_split_radcil_seed7",
        )
    )
    rows.extend(
        _round_rows(
            "results/stage6/adsb_gpcc_cil_seed7_44999118/clustering_results.csv",
            "results/stage6/adsb_gpcc_cil_seed7_44999118/incremental_results.csv",
            "ADS-B",
            "gpcc_radcil_seed7",
        )
    )
    rows.extend(
        _round_rows(
            "results/stage6/lora_reliability_weight_seed7_45048052/reliability_off/clustering_results.csv",
            "results/stage6/lora_reliability_weight_seed7_45048052/reliability_off/incremental_results.csv",
            "LoRa",
            "mvacc_radcil_seed7",
        )
    )
    # LoRa GPCC 只跑 discovery-only，没有增量表；单独放入前端诊断。
    gpcc_lora = _read_csv("results/stage6/lora_gpcc_discovery_gpcc_seed7_44998998/clustering_results.csv")
    for _, item in gpcc_lora.iterrows():
        hungarian = float(item["Hungarian Acc"])
        rows.append(
            {
                "dataset": "LoRa",
                "variant": "gpcc_discovery_only_seed7",
                "round": str(item["Round"]),
                "cluster_count": int(item.get("Final Cluster Count", item.get("Cluster Count", -1))),
                "cluster_count_error": int(item.get("Cluster Count Error", -1)),
                "purity": float(item["Purity"]),
                "ari": float(item["ARI"]),
                "hungarian_acc": hungarian,
                "pseudo_noise_floor": 1.0 - hungarian,
                "overall": None,
                "old": None,
                "new": None,
                "forgetting": None,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """按数据集和变体计算均值，并给出下一步判断。"""
    frame = pd.DataFrame(rows)
    grouped = []
    for (dataset, variant), group in frame.groupby(["dataset", "variant"], sort=True):
        grouped.append(
            {
                "dataset": dataset,
                "variant": variant,
                "mean_hungarian": float(group["hungarian_acc"].mean()),
                "mean_purity": float(group["purity"].mean()),
                "mean_noise_floor": float(group["pseudo_noise_floor"].mean()),
                "r3_hungarian": float(group[group["round"] == "R3"]["hungarian_acc"].iloc[0]),
                "r3_overall": _none_if_nan(group[group["round"] == "R3"]["overall"].iloc[0]),
                "r3_new": _none_if_nan(group[group["round"] == "R3"]["new"].iloc[0]),
            }
        )
    return {
        "rows": rows,
        "grouped": grouped,
        "judgment": {
            "adsb": (
                "ADS-B 的 target split 已把簇数补齐，但 R3 Hungarian 约 0.60，"
                "伪标签噪声下界仍接近 40%；下一步应优先提高发现表征纯度，"
                "而不是继续在 RADCIL 后端加小损失。"
            ),
            "lora": (
                "LoRa 的 R1-R3 Hungarian 大约只有 0.33-0.48，且 R2 仍可能欠簇；"
                "这意味着训练标签本身非常噪，后端改动很难显著拉升 Overall。"
                "下一步应做训练期跨天表征预训练/域不变表征，而不是继续调后端权重。"
            ),
        },
        "next_candidate": (
            "Stage 9 建议转向 discovery 表征重训：用 Day1 train + 当前 discovery 做无标签/弱标签"
            "双视图预训练，再重新抽取 embedding 和聚类；先 discovery-only 过门槛，再跑 CIL。"
        ),
    }


def _none_if_nan(value: Any) -> float | None:
    """把 pandas/float NaN 转成 JSON 里的 null。"""
    if value is None or pd.isna(value):
        return None
    return float(value)


def write_report(summary: dict[str, Any], output: Path, summary_json: Path) -> None:
    """写出 Markdown 和 JSON 诊断报告。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 9 前端瓶颈诊断",
        "",
        "## 结论",
        "",
        f"- ADS-B：{summary['judgment']['adsb']}",
        f"- LoRa：{summary['judgment']['lora']}",
        f"- 下一候选：{summary['next_candidate']}",
        "",
        "## 变体均值",
        "",
        "| Dataset | Variant | Mean Hungarian | Mean Purity | Mean Noise Floor | R3 Hungarian | R3 Overall | R3 New |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["grouped"]:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['mean_hungarian']:.4f} | "
            f"{row['mean_purity']:.4f} | {row['mean_noise_floor']:.4f} | "
            f"{row['r3_hungarian']:.4f} | {_fmt(row['r3_overall'])} | {_fmt(row['r3_new'])} |"
        )
    lines.extend(
        [
            "",
            "## 逐轮证据",
            "",
            "| Dataset | Variant | Round | Clusters | Count Error | Purity | ARI | Hungarian | Noise Floor | Overall | New |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['round']} | {row['cluster_count']} | "
            f"{row['cluster_count_error']} | {row['purity']:.4f} | {row['ari']:.4f} | "
            f"{row['hungarian_acc']:.4f} | {row['pseudo_noise_floor']:.4f} | "
            f"{_fmt(row['overall'])} | {_fmt(row['new'])} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fmt(value: float | None) -> str:
    """格式化可空指标。"""
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def main() -> int:
    """生成前端瓶颈诊断报告。"""
    summary = summarize(collect_rows())
    write_report(
        summary,
        ROOT / "results/stage9/STAGE9_FRONTEND_BOTTLENECK_DIAGNOSIS.md",
        ROOT / "results/stage9/stage9_frontend_bottleneck_diagnosis.json",
    )
    print(json.dumps({"ok": True, "next_candidate": summary["next_candidate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
