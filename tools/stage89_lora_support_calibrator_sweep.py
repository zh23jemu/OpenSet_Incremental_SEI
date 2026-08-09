#!/usr/bin/env python3
"""Stage89：LoRa support=2 旧类校准器 sweep。

Stage88 说明：每个旧类 2 条同日 support recording 时，Stage87 的 observed
Overall 为 0.5133，但按 R3 原始 20 旧类 + 5 新类比例重算后为 0.4976，
距离 50% 只差约 0.003 Old Acc。因此本阶段不再改变 support 数量，也不
改变评估口径，只在 support=2 下比较更稳的旧类校准器，看能否把 adjusted
Overall 真正推过 50%。

边界：
* support 仍从 held-out 旧类 recording 中按真值抽取，所以这是数据需求/
  上限诊断，不是无 support strict 正式成绩。
* support 样本从评估分母剔除；报告同时给 observed 和原始类比例 adjusted。
* 新类预测保持主模型输出，不使用新类真值训练校准器。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.stage68_lora_old_day_oracle_calibration_report import _load_dump  # noqa: E402
from tools.stage69_lora_old_day_calibration_report import _evaluate, _stage_bounds  # noqa: E402
from tools.stage69_lora_old_day_calibration_report import _augment_logits_for_calibration  # noqa: E402
from tools.stage87_lora_same_day_support_upper_bound import (  # noqa: E402
    _normalise_rows,
    _select_support_mask,
)


STAGES = ["initial", "after_r1", "after_r2", "after_r3"]


def _evaluate_subset(
    stage: str,
    y_true: np.ndarray,
    pred: np.ndarray,
    eval_mask: np.ndarray,
    initial_reference: float | None,
) -> dict[str, float]:
    """复用项目指标函数，但先剔除 support 样本。"""

    return _evaluate(stage, y_true[eval_mask], pred[eval_mask], initial_reference)


def _adjusted_overall(row: dict[str, Any], old_classes: int, new_classes: int) -> float:
    """按原始 R3 类比例计算加权 Overall。"""

    total = float(old_classes + new_classes)
    return float(old_classes) / total * float(row["old"]) + float(new_classes) / total * float(row["new"])


def _required_old_for_target(target: float, new_acc: float, old_classes: int, new_classes: int) -> float:
    """固定 New 时达到目标 Overall 所需 Old。"""

    old_w = float(old_classes) / float(old_classes + new_classes)
    new_w = float(new_classes) / float(old_classes + new_classes)
    return (float(target) - new_w * float(new_acc)) / old_w


def _prototype_pred(
    logits: np.ndarray,
    y_true: np.ndarray,
    base_pred: np.ndarray,
    support_mask: np.ndarray,
    eval_mask: np.ndarray,
    old_end: int,
    shrink: float,
) -> np.ndarray:
    """用 support logits 原型重判旧类；shrink 控制类原型向全局旧类中心收缩。"""

    pred = np.asarray(base_pred, dtype=np.int64).copy()
    logits_norm = _normalise_rows(logits)
    old_support = support_mask & (y_true < old_end)
    if not np.any(old_support):
        return pred
    global_center = logits_norm[old_support].mean(axis=0)
    classes: list[int] = []
    prototypes: list[np.ndarray] = []
    for class_id in range(int(old_end)):
        class_mask = support_mask & (y_true == class_id)
        if not np.any(class_mask):
            continue
        class_center = logits_norm[class_mask].mean(axis=0)
        proto = (1.0 - float(shrink)) * class_center + float(shrink) * global_center
        classes.append(class_id)
        prototypes.append(proto)
    if not prototypes:
        return pred
    proto = _normalise_rows(np.vstack(prototypes))
    old_eval = eval_mask & (y_true < old_end)
    if np.any(old_eval):
        nearest = np.argmax(logits_norm[old_eval] @ proto.T, axis=1)
        pred[old_eval] = np.asarray([classes[int(idx)] for idx in nearest], dtype=np.int64)
    return pred


def _fit_predict_logreg(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    base_pred: np.ndarray,
    old_eval: np.ndarray,
    *,
    c_value: float,
    class_weight: str | None,
) -> np.ndarray:
    """训练旧类 logreg 专家，并只改写真实旧类 eval 样本。"""

    pred = np.asarray(base_pred, dtype=np.int64).copy()
    if len(np.unique(train_y)) < 2:
        return pred
    clf = LogisticRegression(
        max_iter=4000,
        solver="lbfgs",
        C=float(c_value),
        class_weight=class_weight,
        n_jobs=1,
    )
    clf.fit(np.asarray(train_x, dtype=np.float32), np.asarray(train_y, dtype=np.int64))
    if np.any(old_eval):
        pred[old_eval] = clf.predict(np.asarray(eval_x[old_eval], dtype=np.float32)).astype(np.int64)
    return pred


def _variant_predictions(
    logits: np.ndarray,
    y_true: np.ndarray,
    base_pred: np.ndarray,
    support_mask: np.ndarray,
    eval_mask: np.ndarray,
    old_end: int,
    c_values: list[float],
) -> dict[str, np.ndarray]:
    """生成本阶段要比较的所有旧类校准器预测。"""

    old_eval = eval_mask & (y_true < old_end)
    support_y = y_true[support_mask].astype(np.int64)
    variants: dict[str, np.ndarray] = {
        "baseline_support2": np.asarray(base_pred, dtype=np.int64).copy(),
        "prototype_shrink0": _prototype_pred(logits, y_true, base_pred, support_mask, eval_mask, old_end, shrink=0.0),
        "prototype_shrink0p10": _prototype_pred(logits, y_true, base_pred, support_mask, eval_mask, old_end, shrink=0.10),
        "prototype_shrink0p25": _prototype_pred(logits, y_true, base_pred, support_mask, eval_mask, old_end, shrink=0.25),
    }

    raw_train = logits[support_mask].astype(np.float32)
    raw_eval = logits.astype(np.float32)
    aug_all = _augment_logits_for_calibration(logits)
    aug_train = aug_all[support_mask].astype(np.float32)
    mean = aug_train.mean(axis=0, keepdims=True)
    std = np.maximum(aug_train.std(axis=0, keepdims=True), 1e-6)
    aug_all = ((aug_all - mean) / std).astype(np.float32)
    for c_value in c_values:
        label = str(c_value).replace(".", "p")
        variants[f"logreg_c{label}"] = _fit_predict_logreg(
            raw_train,
            support_y,
            raw_eval,
            base_pred,
            old_eval,
            c_value=c_value,
            class_weight=None,
        )
        variants[f"logreg_bal_c{label}"] = _fit_predict_logreg(
            raw_train,
            support_y,
            raw_eval,
            base_pred,
            old_eval,
            c_value=c_value,
            class_weight="balanced",
        )
        variants[f"aug_logreg_c{label}"] = _fit_predict_logreg(
            aug_train,
            support_y,
            aug_all,
            base_pred,
            old_eval,
            c_value=c_value,
            class_weight=None,
        )
    return variants


def _collect(save_dir: Path, support_count: int, seed: int, c_values: list[float]) -> dict[str, Any]:
    """按 support_count 收集逐轮结果。"""

    rows: list[dict[str, Any]] = []
    initial_refs: dict[str, float] = {}
    for stage in STAGES:
        dump = _load_dump(save_dir, stage)
        y_true = np.asarray(dump["y_true"], dtype=np.int64)
        logits = np.asarray(dump["logits"], dtype=np.float32)
        base_pred = np.asarray(dump["pred_mapped"], dtype=np.int64)
        recording_id = np.asarray(dump["recording_id"]) if "recording_id" in dump else None
        old_end = int(_stage_bounds(stage)["old_end"])
        if stage == "initial":
            support_mask = np.zeros(len(y_true), dtype=bool)
        else:
            support_mask = _select_support_mask(
                y_true=y_true,
                recording_id=recording_id,
                old_end=old_end,
                recordings_per_old_class=support_count,
                seed=seed + int(stage.replace("after_r", "")) * 17,
            )
        eval_mask = ~support_mask
        variants = _variant_predictions(logits, y_true, base_pred, support_mask, eval_mask, old_end, c_values)
        for variant, pred in variants.items():
            ref = None if stage == "initial" else initial_refs.get(variant)
            metrics = _evaluate_subset(stage, y_true, pred, eval_mask, ref)
            if stage == "initial":
                initial_refs[variant] = float(metrics["initial_known"])
            row = {
                "support_recordings_per_old_class": int(support_count),
                "variant": variant,
                **metrics,
                "support_samples": int(np.sum(support_mask)),
                "eval_samples": int(np.sum(eval_mask)),
            }
            if stage == "After R3":
                row["adjusted_original_mix_overall"] = _adjusted_overall(row, old_classes=20, new_classes=5)
                row["required_old_for_50"] = _required_old_for_target(0.50, row["new"], old_classes=20, new_classes=5)
                row["old_gap_to_50"] = float(row["old"]) - float(row["required_old_for_50"])
                row["passes_adjusted_50"] = bool(float(row["adjusted_original_mix_overall"]) >= 0.50)
            rows.append(row)
    r3_rows = [row for row in rows if row["stage"] == "After R3"]
    best_adjusted = max(
        r3_rows,
        key=lambda row: (
            float(row["adjusted_original_mix_overall"]),
            float(row["old"]),
            float(row["new"]),
        ),
    )
    return {"rows": rows, "r3_rows": r3_rows, "best_adjusted_r3": best_adjusted}


def _fmt(value: Any) -> str:
    """统一报告格式。"""

    return f"{float(value):.4f}"


def main() -> int:
    """生成 Stage89 sweep 报告。"""

    parser = argparse.ArgumentParser(description="Stage89 LoRa support=2 calibrator sweep")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--support-count", type=int, default=2)
    parser.add_argument("--c-values", default="0.03,0.1,0.3,1.0,3.0,10.0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    c_values = [float(item) for item in str(args.c_values).split(",") if item.strip()]
    result = _collect(Path(args.save_dir), int(args.support_count), int(args.seed), c_values)
    best = result["best_adjusted_r3"]
    passes = bool(best["passes_adjusted_50"])

    lines = [
        f"# Stage89 LoRa support=2 校准器 sweep（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'原始类比例 adjusted Overall 已过 50%。' if passes else '原始类比例 adjusted Overall 仍未过 50%。'}",
        (
            f"- 最佳 R3：`{best['variant']}`，Adjusted Overall/Old/New="
            f"`{_fmt(best['adjusted_original_mix_overall'])}/{_fmt(best['old'])}/{_fmt(best['new'])}`，"
            f"Old gap to 50=`{_fmt(best['old_gap_to_50'])}`。"
        ),
        "- 边界：support 仍来自 held-out 旧类真值；本阶段是 support 数据需求诊断，不是无 support strict 成绩。",
        "",
        "## R3 Sweep",
        "",
        "| Variant | Observed Overall | Adjusted Overall | Old | New | Forgetting | Required Old @50 | Old Gap | Pass Adjusted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(result["r3_rows"], key=lambda item: float(item["adjusted_original_mix_overall"]), reverse=True):
        lines.append(
            "| {variant} | {observed} | {adjusted} | {old} | {new} | {forgetting} | {required_old} | {old_gap} | {passed} |".format(
                variant=row["variant"],
                observed=_fmt(row["overall"]),
                adjusted=_fmt(row["adjusted_original_mix_overall"]),
                old=_fmt(row["old"]),
                new=_fmt(row["new"]),
                forgetting=_fmt(row["forgetting"]),
                required_old=_fmt(row["required_old_for_50"]),
                old_gap=_fmt(row["old_gap_to_50"]),
                passed=str(bool(row["passes_adjusted_50"])),
            )
        )
    lines.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage89_lora_support_calibrator_sweep_v1",
                "job_id": str(args.job_id),
                "save_dir": str(args.save_dir),
                "support_count": int(args.support_count),
                "c_values": c_values,
                "result": result,
                "passes_adjusted_50": passes,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Stage89 报告：{output}")
    print(json.dumps({"best_adjusted_r3": best, "passes_adjusted_50": passes}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
