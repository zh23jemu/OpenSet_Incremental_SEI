#!/usr/bin/env python3
"""Stage66：LoRa 50% 目标缺口分解报告。

这个脚本不训练模型，也不读取 held-out eval 的逐样本预测做调参；它只汇总
已经完成实验的 R3 聚合指标，用公开协议中的旧/新类数量权重拆解 Overall。
核心目的是回答：当前 LoRa 为什么离 50% 远，以及继续攻应该优先救 Old 还是 New。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_r3(csv_path: Path) -> pd.Series:
    """读取指标 CSV 的 R3 行；兼容 Stage/Round 两种列名。"""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    frame = pd.read_csv(csv_path)
    if "Stage" in frame.columns:
        rows = frame.loc[frame["Stage"].astype(str) == "After R3"]
        return rows.iloc[-1] if not rows.empty else frame.tail(1).iloc[0]
    if "Round" in frame.columns:
        rows = frame.loc[frame["Round"].astype(str) == "R3"]
        return rows.iloc[-1] if not rows.empty else frame.tail(1).iloc[0]
    return frame.tail(1).iloc[0]


def _scenario(label: str, save_dir: Path, seed: int, known_classes: int = 10, round_size: int = 5, rounds: int = 3) -> dict:
    """把一个实验目录转成 R3 缺口分解记录。

    R3 时旧类数量为初始已知类 + 前两轮新增类，新类数量为第三轮新增类。
    在当前 LoRa 10+5x3 协议下，R3 是 20 个旧类和 5 个新类，所以 Old 权重是 0.8。
    """
    row = _read_r3(save_dir / "incremental_results.csv")
    old_classes = known_classes + round_size * (rounds - 1)
    new_classes = round_size
    total_classes = old_classes + new_classes
    old_weight = old_classes / total_classes
    new_weight = new_classes / total_classes
    old_acc = float(row["Old Acc"])
    new_acc = float(row["New Acc"])
    overall = float(row["Overall Acc"])

    # 若保持 New 不变，要达到 50% Overall 需要的 Old；反过来也计算 New。
    required_old_for_50 = max(0.0, min(1.0, (0.50 - new_weight * new_acc) / old_weight))
    required_new_for_50 = max(0.0, min(1.0, (0.50 - old_weight * old_acc) / new_weight))
    max_overall_if_new_perfect = old_weight * old_acc + new_weight
    max_overall_if_old_perfect = old_weight + new_weight * new_acc

    return {
        "label": label,
        "seed": seed,
        "save_dir": str(save_dir),
        "overall": overall,
        "old_acc": old_acc,
        "new_acc": new_acc,
        "forgetting": float(row["Forgetting Rate"]),
        "old_weight": old_weight,
        "new_weight": new_weight,
        "required_old_for_50_if_new_fixed": required_old_for_50,
        "required_new_for_50_if_old_fixed": required_new_for_50,
        "max_overall_if_new_perfect": max_overall_if_new_perfect,
        "max_overall_if_old_perfect": max_overall_if_old_perfect,
    }


def _fmt(x: float) -> str:
    """把浮点指标统一格式化成 4 位小数。"""
    return f"{x:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage66 LoRa target gap decomposition")
    parser.add_argument("--root", default="results")
    parser.add_argument("--output", default="results/stage66/STAGE66_LORA_TARGET_GAP_DECOMPOSITION_REPORT.md")
    parser.add_argument("--summary-json", default="results/stage66/stage66_lora_target_gap_decomposition_summary.json")
    args = parser.parse_args()

    root = Path(args.root)
    records = [
        _scenario("Stage48 raw s28 rec065", root / "stage48" / f"lora_s28_rec065_seed{seed}_46124799", seed)
        for seed in (7, 13, 31)
    ]
    records.append(
        _scenario(
            "Stage63 consistency pretrain",
            root / "stage63" / "lora_s28_consistency_seed7_46319406",
            7,
        )
    )
    records.append(
        _scenario(
            "Stage64 masked reconstruction",
            root / "stage64" / "lora_s28_masked_recon_seed7_46321233",
            7,
        )
    )

    best = max(records, key=lambda item: float(item["overall"]))
    stage48_records = [item for item in records if item["label"] == "Stage48 raw s28 rec065"]
    stage48_old_mean = sum(float(item["old_acc"]) for item in stage48_records) / len(stage48_records)
    stage48_new_mean = sum(float(item["new_acc"]) for item in stage48_records) / len(stage48_records)
    stage48_if_new_perfect = sum(float(item["max_overall_if_new_perfect"]) for item in stage48_records) / len(stage48_records)
    stage48_required_old = sum(float(item["required_old_for_50_if_new_fixed"]) for item in stage48_records) / len(stage48_records)

    lines = [
        "# Stage66 LoRa 50% 目标缺口分解报告",
        "",
        "## 结论",
        "",
        "- LoRa 低分的主因已经不是单纯 New 或聚类；R3 Overall 被 20 个旧类强烈主导，Old 权重约 0.80。",
        f"- Stage48 三种子 Old 均值只有 {_fmt(stage48_old_mean)}，New 均值为 {_fmt(stage48_new_mean)}；即使把 New 提到 100%，Overall 理论上也只有约 {_fmt(stage48_if_new_perfect)}。",
        f"- 如果保持当前 New 水平，Old 平均需要提高到约 {_fmt(stage48_required_old)}，Overall 才能到 50%。这比当前 Old 高约 {_fmt(stage48_required_old - stage48_old_mean)}。",
        f"- 当前最好单点是 {best['label']} seed {best['seed']}，Overall={_fmt(float(best['overall']))}；距离 50% 仍差 {_fmt(0.50 - float(best['overall']))}。",
        "- 因此继续冲 50 的真正风险点是旧类跨天保持能力，而不是继续把新类 New 从 0.48 小幅推到 0.52。",
        "",
        "## R3 缺口表",
        "",
        "| Label | Seed | Overall | Old | New | Forgetting | Old Weight | New Weight | Need Old for 50% if New Fixed | Need New for 50% if Old Fixed | Max Overall if New=100% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in records:
        lines.append(
            "| {label} | {seed} | {overall} | {old} | {new} | {forgetting} | {old_w} | {new_w} | {need_old} | {need_new} | {max_new} |".format(
                label=item["label"],
                seed=item["seed"],
                overall=_fmt(float(item["overall"])),
                old=_fmt(float(item["old_acc"])),
                new=_fmt(float(item["new_acc"])),
                forgetting=_fmt(float(item["forgetting"])),
                old_w=_fmt(float(item["old_weight"])),
                new_w=_fmt(float(item["new_weight"])),
                need_old=_fmt(float(item["required_old_for_50_if_new_fixed"])),
                need_new=_fmt(float(item["required_new_for_50_if_old_fixed"])),
                max_new=_fmt(float(item["max_overall_if_new_perfect"])),
            )
        )
    lines.extend(
        [
            "",
            "## 下一步判断",
            "",
            "- 若继续做算法，应优先尝试旧类跨天保持的大结构，例如按 Day1/Day2-4 做显式域泛化或旧类 replay 表征再训练。",
            "- 若继续按当前 strict symbol-level 协议汇报，LoRa 只能写成已从约 15%-17% 提升到约 28%-30%，但没有达到 50%。",
            "",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage66_lora_target_gap_decomposition_v1",
                "records": records,
                "best_overall": best,
                "stage48_old_mean": stage48_old_mean,
                "stage48_new_mean": stage48_new_mean,
                "stage48_max_overall_if_new_perfect": stage48_if_new_perfect,
                "stage48_required_old_for_50_if_new_fixed": stage48_required_old,
                "diagnosis": "OLD_CLASS_CROSS_DAY_RETENTION_IS_PRIMARY_TARGET_GAP",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
