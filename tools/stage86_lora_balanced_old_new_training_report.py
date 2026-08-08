"""汇总 Stage86 LoRa 旧类跨天监督 + 新类 margin 平衡训练结果。

Stage86 不引入新的数据划分或评估口径，只把两个已验证为“单边有效但会伤害另一侧”
的训练约束放在同一轮 CIL 里，并用更低权重做保守平衡：

* 旧类跨天监督：来自 Day2-4 旧设备 IQ_1-7，严格不含 held-out eval 的 IQ_8-10。
* 新类 margin：只对当前轮 discovery 伪新类样本生效，不读取未知真值。

报告以 symbol-level R3 为正式判断口径，recording-level 仅作诊断。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


STAGE46_SEED7_BASELINE = {
    "overall": 0.3005,
    "old": 0.2429,
    "new": 0.5310,
    "forgetting": 0.1155,
}

STAGE76_SINGLE_SIDED_OLD = {
    "overall": 0.2886,
    "old": 0.2601,
    "new": 0.4024,
    "forgetting": 0.1631,
}

STAGE85_SINGLE_SIDED_NEW = {
    "overall": 0.2676,
    "old": 0.1911,
    "new": 0.5738,
    "forgetting": 0.2321,
}


def _read_r3_metrics(save_dir: Path) -> dict[str, float]:
    """读取主实验 incremental_results.csv 的 After R3 指标。"""

    csv_path = save_dir / "incremental_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing incremental results: {csv_path}")
    frame = pd.read_csv(csv_path)
    rows = frame[frame["Stage"].astype(str).str.strip().str.lower().isin({"after r3", "r3"})]
    row = rows.iloc[-1] if not rows.empty else frame.iloc[-1]
    return {
        "overall": float(row.get("Overall Acc", 0.0)),
        "old": float(row.get("Old Acc", 0.0)),
        "new": float(row.get("New Acc", 0.0)),
        "forgetting": float(row.get("Forgetting Rate", 0.0)),
        "macro_f1": float(row.get("Macro F1", 0.0)),
    }


def _read_recording_r3(save_dir: Path) -> dict[str, float] | None:
    """读取 recording-level 诊断指标；缺失时返回 None。"""

    csv_path = save_dir / "recording_level_incremental_results.csv"
    if not csv_path.exists():
        return None
    frame = pd.read_csv(csv_path)
    rows = frame[frame["Stage"].astype(str).str.strip().str.lower().isin({"after r3", "r3"})]
    row = rows.iloc[-1] if not rows.empty else frame.iloc[-1]
    return {
        "recording_overall": float(row.get("Overall Acc", 0.0)),
        "recording_old": float(row.get("Old Acc", 0.0)),
        "recording_new": float(row.get("New Acc", 0.0)),
    }


def _delta(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    """按统一键计算候选相对某个 baseline 的指标差异。"""

    return {
        key: float(metrics[key]) - float(baseline[key])
        for key in ("overall", "old", "new", "forgetting")
    }


def _fmt_metric(metrics: dict[str, float], key: str) -> str:
    """把缺失风险降到最低的指标格式化辅助函数。"""

    return f"{float(metrics[key]):.4f}"


def main() -> int:
    """生成 Stage86 seed7 报告和 JSON 摘要。"""

    parser = argparse.ArgumentParser(description="Stage86 LoRa balanced old/new training seed7 报告器")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--old-day-ce-weight", type=float, required=True)
    parser.add_argument("--old-day-feature-weight", type=float, required=True)
    parser.add_argument("--margin-weight", type=float, required=True)
    parser.add_argument("--margin", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    metrics = _read_r3_metrics(save_dir)
    recording_metrics = _read_recording_r3(save_dir)
    delta_vs_stage46 = _delta(metrics, STAGE46_SEED7_BASELINE)
    delta_vs_stage76 = _delta(metrics, STAGE76_SINGLE_SIDED_OLD)
    delta_vs_stage85 = _delta(metrics, STAGE85_SINGLE_SIDED_NEW)

    # 通过门槛保持偏严格：必须超过 Stage46 seed7 Overall，并且 Old/New 都不能明显塌缩。
    # Forgetting 不要求优于 Stage46，但不能比 Stage46 明显恶化太多。
    passed = (
        delta_vs_stage46["overall"] > 0.0
        and delta_vs_stage46["old"] >= -0.02
        and delta_vs_stage46["new"] >= -0.02
        and delta_vs_stage46["forgetting"] <= 0.05
    )

    recording_cells = (
        f"{recording_metrics['recording_overall']:.4f} | "
        f"{recording_metrics['recording_old']:.4f} | "
        f"{recording_metrics['recording_new']:.4f}"
        if recording_metrics
        else "- | - | -"
    )

    lines = [
        f"# Stage86 LoRa 新旧平衡训练 seed7 报告（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'通过 seed7 门槛，可扩 seed13/31。' if passed else '未通过 seed7 门槛，暂不扩三种子。'}",
        (
            "- 配置：old-day CE "
            f"`{args.old_day_ce_weight}`，old-day feature `{args.old_day_feature_weight}`，"
            f"margin weight `{args.margin_weight}`，margin `{args.margin}`。"
        ),
        "- 改动范围：保持 Stage48/Stage46 的 raw-s28、Chirp、recording-consensus 0.65、LoRa SSL、joint refinement 和 RADCIL replay 设置，只叠加低权重旧类跨天监督与新类 margin。",
        "",
        "## R3 指标",
        "",
        "| Variant | Overall | Old | New | Forgetting | Macro F1 | Recording Overall | Recording Old | Recording New |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| stage46_raw_s28_seed7 | {_fmt_metric(STAGE46_SEED7_BASELINE, 'overall')} | "
            f"{_fmt_metric(STAGE46_SEED7_BASELINE, 'old')} | {_fmt_metric(STAGE46_SEED7_BASELINE, 'new')} | "
            f"{_fmt_metric(STAGE46_SEED7_BASELINE, 'forgetting')} | n/a | 0.3067 | n/a | 0.6667 |"
        ),
        (
            f"| stage76_old_day_joint_cil | {_fmt_metric(STAGE76_SINGLE_SIDED_OLD, 'overall')} | "
            f"{_fmt_metric(STAGE76_SINGLE_SIDED_OLD, 'old')} | {_fmt_metric(STAGE76_SINGLE_SIDED_OLD, 'new')} | "
            f"{_fmt_metric(STAGE76_SINGLE_SIDED_OLD, 'forgetting')} | n/a | n/a | n/a | n/a |"
        ),
        (
            f"| stage85_new_old_margin | {_fmt_metric(STAGE85_SINGLE_SIDED_NEW, 'overall')} | "
            f"{_fmt_metric(STAGE85_SINGLE_SIDED_NEW, 'old')} | {_fmt_metric(STAGE85_SINGLE_SIDED_NEW, 'new')} | "
            f"{_fmt_metric(STAGE85_SINGLE_SIDED_NEW, 'forgetting')} | n/a | n/a | n/a | n/a |"
        ),
        (
            f"| stage86_balanced_old_new | {metrics['overall']:.4f} | {metrics['old']:.4f} | "
            f"{metrics['new']:.4f} | {metrics['forgetting']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{recording_cells} |"
        ),
        "",
        "## Delta vs Stage46 Seed7",
        "",
        f"- Overall: {delta_vs_stage46['overall']:+.4f}",
        f"- Old: {delta_vs_stage46['old']:+.4f}",
        f"- New: {delta_vs_stage46['new']:+.4f}",
        f"- Forgetting: {delta_vs_stage46['forgetting']:+.4f}",
        "",
        "## 单边候选对照",
        "",
        f"- vs Stage76：Overall {delta_vs_stage76['overall']:+.4f}，Old {delta_vs_stage76['old']:+.4f}，New {delta_vs_stage76['new']:+.4f}，Forgetting {delta_vs_stage76['forgetting']:+.4f}",
        f"- vs Stage85：Overall {delta_vs_stage85['overall']:+.4f}，Old {delta_vs_stage85['old']:+.4f}，New {delta_vs_stage85['new']:+.4f}，Forgetting {delta_vs_stage85['forgetting']:+.4f}",
        "",
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage86_lora_balanced_old_new_training_seed7_summary_v1",
                "job_id": str(args.job_id),
                "save_dir": str(save_dir),
                "weights": {
                    "old_day_ce_weight": float(args.old_day_ce_weight),
                    "old_day_feature_weight": float(args.old_day_feature_weight),
                    "margin_weight": float(args.margin_weight),
                    "margin": float(args.margin),
                },
                "stage46_seed7_baseline": STAGE46_SEED7_BASELINE,
                "stage76_single_sided_old": STAGE76_SINGLE_SIDED_OLD,
                "stage85_single_sided_new": STAGE85_SINGLE_SIDED_NEW,
                "metrics": metrics,
                "recording_metrics": recording_metrics,
                "delta_vs_stage46_seed7": delta_vs_stage46,
                "delta_vs_stage76": delta_vs_stage76,
                "delta_vs_stage85": delta_vs_stage85,
                "passed_seed7_gate": bool(passed),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Stage86 报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
