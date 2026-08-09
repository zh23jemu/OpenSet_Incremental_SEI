#!/usr/bin/env python3
"""Stage90：LoRa support=2 校准稳定性 repeat。

Stage89 已证明在 seed7 模型 dump 上，support=2 且 old-class logreg C=0.3
可以把按原始 20旧类 + 5新类比例重算后的 R3 Overall 推到 0.5105。
但该结果依赖从 held-out 旧类样本里抽取 support，因此还需要确认：

1. 换不同 support 抽样种子后，结果是否仍稳定过 50%；
2. 最坏/均值口径是否仍能支撑“少量同日旧类 support 可以打开 50% 缺口”；
3. 这个结论仍明确属于额外同日旧类标注的数据需求/上限诊断，而不是无 support strict 成绩。

本脚本只读取 Stage68 的 end_to_end_eval_dumps，不训练新模型，不读取新类真值做校准。
support 样本仍由旧类 held-out 真值抽取，因此必须作为诊断结果单独汇报。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.stage68_lora_old_day_oracle_calibration_report import _load_dump  # noqa: E402
from tools.stage69_lora_old_day_calibration_report import _evaluate, _stage_bounds  # noqa: E402
from tools.stage87_lora_same_day_support_upper_bound import _select_support_mask  # noqa: E402
from tools.stage89_lora_support_calibrator_sweep import (  # noqa: E402
    _adjusted_overall,
    _fit_predict_logreg,
    _required_old_for_target,
)


STAGES = ["initial", "after_r1", "after_r2", "after_r3"]


def _evaluate_subset(
    stage: str,
    y_true: np.ndarray,
    pred: np.ndarray,
    eval_mask: np.ndarray,
    initial_reference: float | None,
) -> dict[str, float]:
    """剔除 support 后复用项目统一指标函数。"""

    return _evaluate(stage, y_true[eval_mask], pred[eval_mask], initial_reference)


def _logreg_c03_pred(
    logits: np.ndarray,
    y_true: np.ndarray,
    base_pred: np.ndarray,
    support_mask: np.ndarray,
    eval_mask: np.ndarray,
    old_end: int,
    *,
    c_value: float,
) -> np.ndarray:
    """只运行 Stage89 最佳 old-class logreg 校准器。

    为保持和 Stage89/Stage87 可比，输入特征直接使用原始 logits，solver 和
    max_iter 由 Stage89 的 `_fit_predict_logreg` 统一控制。校准器只改写真实旧类
    eval 样本；真实新类样本保持主模型预测，用于观察 old support 是否会伤害 New。
    """

    old_eval = eval_mask & (y_true < old_end)
    if not np.any(support_mask):
        return np.asarray(base_pred, dtype=np.int64).copy()
    return _fit_predict_logreg(
        logits[support_mask].astype(np.float32),
        y_true[support_mask].astype(np.int64),
        logits.astype(np.float32),
        base_pred,
        old_eval,
        c_value=c_value,
        class_weight=None,
    )


def _collect_one_repeat(
    save_dir: Path,
    support_seed: int,
    support_count: int,
    c_value: float,
) -> dict[str, Any]:
    """收集一个 support 抽样种子的逐轮指标。"""

    rows: list[dict[str, Any]] = []
    initial_reference: float | None = None
    for stage_key in STAGES:
        dump = _load_dump(save_dir, stage_key)
        y_true = np.asarray(dump["y_true"], dtype=np.int64)
        logits = np.asarray(dump["logits"], dtype=np.float32)
        base_pred = np.asarray(dump["pred_mapped"], dtype=np.int64)
        recording_id = np.asarray(dump["recording_id"]) if "recording_id" in dump else None
        old_end = int(_stage_bounds(stage_key)["old_end"])

        if stage_key == "initial":
            support_mask = np.zeros(len(y_true), dtype=bool)
        else:
            round_index = int(stage_key.replace("after_r", ""))
            support_mask = _select_support_mask(
                y_true=y_true,
                recording_id=recording_id,
                old_end=old_end,
                recordings_per_old_class=support_count,
                seed=int(support_seed) + round_index * 17,
            )
        eval_mask = ~support_mask
        pred = _logreg_c03_pred(
            logits=logits,
            y_true=y_true,
            base_pred=base_pred,
            support_mask=support_mask,
            eval_mask=eval_mask,
            old_end=old_end,
            c_value=c_value,
        )
        metrics = _evaluate_subset(stage_key, y_true, pred, eval_mask, initial_reference)
        if stage_key == "initial":
            initial_reference = float(metrics["initial_known"])

        row: dict[str, Any] = {
            "support_seed": int(support_seed),
            "support_recordings_per_old_class": int(support_count),
            "variant": f"logreg_c{str(c_value).replace('.', 'p')}",
            **metrics,
            "support_samples": int(np.sum(support_mask)),
            "eval_samples": int(np.sum(eval_mask)),
        }
        if row["stage"] == "After R3":
            row["adjusted_original_mix_overall"] = _adjusted_overall(row, old_classes=20, new_classes=5)
            row["required_old_for_50"] = _required_old_for_target(0.50, row["new"], old_classes=20, new_classes=5)
            row["old_gap_to_50"] = float(row["old"]) - float(row["required_old_for_50"])
            row["passes_adjusted_50"] = bool(float(row["adjusted_original_mix_overall"]) >= 0.50)
        rows.append(row)

    r3 = next(row for row in rows if row["stage"] == "After R3")
    return {"support_seed": int(support_seed), "rows": rows, "r3": r3}


def _summarise_repeats(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 repeat 的均值、最差值和过线比例。"""

    r3_rows = [item["r3"] for item in repeats]
    adjusted = np.asarray([float(row["adjusted_original_mix_overall"]) for row in r3_rows], dtype=np.float64)
    old = np.asarray([float(row["old"]) for row in r3_rows], dtype=np.float64)
    new = np.asarray([float(row["new"]) for row in r3_rows], dtype=np.float64)
    pass_flags = np.asarray([bool(row["passes_adjusted_50"]) for row in r3_rows], dtype=bool)
    best = max(r3_rows, key=lambda row: float(row["adjusted_original_mix_overall"]))
    worst = min(r3_rows, key=lambda row: float(row["adjusted_original_mix_overall"]))
    return {
        "repeat_count": int(len(r3_rows)),
        "pass_count": int(np.sum(pass_flags)),
        "pass_rate": float(np.mean(pass_flags)),
        "adjusted_mean": float(np.mean(adjusted)),
        "adjusted_std": float(np.std(adjusted, ddof=0)),
        "adjusted_min": float(np.min(adjusted)),
        "adjusted_max": float(np.max(adjusted)),
        "old_mean": float(np.mean(old)),
        "new_mean": float(np.mean(new)),
        "best_r3": best,
        "worst_r3": worst,
    }


def _fmt(value: Any) -> str:
    """统一四位小数格式。"""

    return f"{float(value):.4f}"


def main() -> int:
    """生成 Stage90 repeat 报告。"""

    parser = argparse.ArgumentParser(description="Stage90 LoRa support repeat stability")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--support-count", type=int, default=2)
    parser.add_argument("--support-seeds", default="7,13,31,57,101")
    parser.add_argument("--c-value", type=float, default=0.3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    support_seeds = [int(item) for item in str(args.support_seeds).split(",") if item.strip()]
    repeats = [
        _collect_one_repeat(
            save_dir=Path(args.save_dir),
            support_seed=seed,
            support_count=int(args.support_count),
            c_value=float(args.c_value),
        )
        for seed in support_seeds
    ]
    summary = _summarise_repeats(repeats)
    stable_pass = bool(summary["pass_count"] == summary["repeat_count"])

    lines = [
        f"# Stage90 LoRa support=2 repeat 稳定性诊断（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        (
            f"- 当前判定：{'全部 support repeat 均过 50%。' if stable_pass else 'support repeat 并非全部稳定过 50%。'}"
        ),
        (
            f"- R3 Adjusted Overall mean/std/min/max="
            f"`{_fmt(summary['adjusted_mean'])}/{_fmt(summary['adjusted_std'])}/"
            f"{_fmt(summary['adjusted_min'])}/{_fmt(summary['adjusted_max'])}`，"
            f"pass rate=`{summary['pass_count']}/{summary['repeat_count']}`。"
        ),
        (
            f"- 最差 repeat：support_seed=`{summary['worst_r3']['support_seed']}`，"
            f"Adjusted Overall/Old/New=`{_fmt(summary['worst_r3']['adjusted_original_mix_overall'])}/"
            f"{_fmt(summary['worst_r3']['old'])}/{_fmt(summary['worst_r3']['new'])}`。"
        ),
        "- 边界：support 仍来自 held-out 旧类真值；这是额外同日旧类 support 的上限/数据需求诊断，不是无 support strict 成绩。",
        "",
        "## R3 Repeat",
        "",
        "| Support Seed | Adjusted Overall | Observed Overall | Old | New | Forgetting | Required Old @50 | Old Gap | Pass | Support Samples | Eval Samples |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [item["r3"] for item in repeats]:
        lines.append(
            "| {seed} | {adjusted} | {observed} | {old} | {new} | {forgetting} | {required_old} | {old_gap} | {passed} | {support_samples} | {eval_samples} |".format(
                seed=int(row["support_seed"]),
                adjusted=_fmt(row["adjusted_original_mix_overall"]),
                observed=_fmt(row["overall"]),
                old=_fmt(row["old"]),
                new=_fmt(row["new"]),
                forgetting=_fmt(row["forgetting"]),
                required_old=_fmt(row["required_old_for_50"]),
                old_gap=_fmt(row["old_gap_to_50"]),
                passed=str(bool(row["passes_adjusted_50"])),
                support_samples=int(row["support_samples"]),
                eval_samples=int(row["eval_samples"]),
            )
        )
    lines.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage90_lora_support_repeat_v1",
                "job_id": str(args.job_id),
                "save_dir": str(args.save_dir),
                "support_count": int(args.support_count),
                "support_seeds": support_seeds,
                "c_value": float(args.c_value),
                "summary": summary,
                "repeats": repeats,
                "stable_pass": stable_pass,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Stage90 报告：{output}")
    print(json.dumps({"summary": summary, "stable_pass": stable_pass}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
