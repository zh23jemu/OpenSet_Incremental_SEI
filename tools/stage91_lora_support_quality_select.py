#!/usr/bin/env python3
"""Stage91：LoRa support=2 代表性/质量选择诊断。

Stage90 证明随机 support=2 不稳定：同一模型、同一 logreg C=0.3，
5 个 support 抽样 repeat 只有 1 个过 50%。本阶段继续沿这个风险点往下攻：
不再随机抽 support，而是在每个旧类的候选 recording 池中按不同质量/代表性
启发式选择 2 条 recording，再看原始 20旧类 + 5新类比例下的 R3 Adjusted
Overall 是否更稳定。

重要边界：
* support 候选池仍来自 held-out 旧类真值，因此这是额外同日旧类标注场景的
  上限/数据需求诊断，不是无 support strict 正式成绩。
* 某些策略（如 hard_*）会倾向把困难样本放进 support 并从评估分母剔除，
  可能高估真实部署收益；报告中必须单独解释。
* 本脚本只读取 Stage68 eval dump，不训练新 backbone，不读取新类真值训练校准器。
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
from tools.stage89_lora_support_calibrator_sweep import (  # noqa: E402
    _adjusted_overall,
    _fit_predict_logreg,
    _required_old_for_target,
)


STAGES = ["initial", "after_r1", "after_r2", "after_r3"]
STRATEGIES = [
    "top_true_logit",
    "top_true_margin",
    "low_entropy",
    "centroid_medoid",
    "centroid_diverse",
    "hard_low_margin",
]


def _softmax(x: np.ndarray) -> np.ndarray:
    """稳定 softmax，用于计算候选 recording 的置信度/熵。"""

    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.maximum(exp_x.sum(axis=1, keepdims=True), 1e-12)


def _normalise_rows(x: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化，供类中心 medoid 选择使用。"""

    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def _recording_table(
    logits: np.ndarray,
    y_true: np.ndarray,
    recording_id: np.ndarray | None,
    class_id: int,
    old_end: int,
) -> list[dict[str, Any]]:
    """聚合某个旧类下每条 recording 的质量特征。

    返回的每一行代表一个候选 support recording。所有指标只由该旧类候选
    recording 的 logits 和已知旧类标签得到，不使用新类真值做选择。
    """

    class_idx = np.flatnonzero(y_true == class_id)
    if class_idx.size == 0:
        return []
    if recording_id is None:
        # 极端兼容路径：如果 dump 没有 recording_id，则每个样本退化成一个候选。
        rec_values = np.arange(class_idx.size)
        rec_for_samples = rec_values
    else:
        rec_for_samples = np.asarray(recording_id)[class_idx]
        rec_values = np.unique(rec_for_samples)

    rows: list[dict[str, Any]] = []
    for rec in rec_values:
        if recording_id is None:
            sample_idx = class_idx[[int(rec)]]
        else:
            sample_idx = class_idx[rec_for_samples == rec]
        rec_logits = np.asarray(logits[sample_idx], dtype=np.float64)
        mean_logits = rec_logits.mean(axis=0)
        old_logits = mean_logits[:old_end]
        other_old = np.delete(old_logits, class_id) if old_end > 1 else old_logits
        true_logit = float(mean_logits[class_id])
        true_margin = float(true_logit - np.max(other_old))
        prob = _softmax(rec_logits).mean(axis=0)
        entropy = float(-np.sum(prob * np.log(np.maximum(prob, 1e-12))))
        rows.append(
            {
                "recording": rec.item() if hasattr(rec, "item") else rec,
                "sample_idx": sample_idx,
                "mean_logits": mean_logits,
                "true_logit": true_logit,
                "true_margin": true_margin,
                "entropy": entropy,
            }
        )
    return rows


def _select_recordings(rows: list[dict[str, Any]], support_count: int, strategy: str) -> list[dict[str, Any]]:
    """按给定策略从同类候选 recording 中选择 support。

    策略说明：
    * top_true_logit：选择模型对真实旧类 logit 最高的 recording；
    * top_true_margin：选择真实旧类相对其它旧类 margin 最高的 recording；
    * low_entropy：选择模型整体预测熵最低的 recording；
    * centroid_medoid：选择最接近同类候选池中心的 recording；
    * centroid_diverse：先选 medoid，再选与第一个 medoid 距离较远但仍接近中心的样本；
    * hard_low_margin：选择 margin 最低的困难样本，主要用于诊断，可能高估收益。
    """

    if not rows:
        return []
    k = min(int(support_count), len(rows))
    if strategy == "top_true_logit":
        return sorted(rows, key=lambda row: float(row["true_logit"]), reverse=True)[:k]
    if strategy == "top_true_margin":
        return sorted(rows, key=lambda row: float(row["true_margin"]), reverse=True)[:k]
    if strategy == "low_entropy":
        return sorted(rows, key=lambda row: float(row["entropy"]))[:k]
    if strategy == "hard_low_margin":
        return sorted(rows, key=lambda row: float(row["true_margin"]))[:k]

    feats = _normalise_rows(np.vstack([row["mean_logits"] for row in rows]))
    centroid = _normalise_rows(feats.mean(axis=0, keepdims=True))[0]
    center_distance = np.linalg.norm(feats - centroid[None, :], axis=1)
    if strategy == "centroid_medoid":
        order = np.argsort(center_distance)
        return [rows[int(idx)] for idx in order[:k]]
    if strategy == "centroid_diverse":
        first = int(np.argmin(center_distance))
        selected = [first]
        while len(selected) < k:
            remaining = [idx for idx in range(len(rows)) if idx not in selected]
            # 仍优先靠近类中心，但加入与已选 support 的距离，避免两条 support 过于重复。
            pair_distance = np.min(
                np.linalg.norm(feats[remaining, None, :] - feats[np.asarray(selected)][None, :, :], axis=2),
                axis=1,
            )
            score = -center_distance[remaining] + 0.25 * pair_distance
            selected.append(int(remaining[int(np.argmax(score))]))
        return [rows[idx] for idx in selected]
    raise ValueError(f"Unknown support strategy: {strategy}")


def _quality_support_mask(
    logits: np.ndarray,
    y_true: np.ndarray,
    recording_id: np.ndarray | None,
    old_end: int,
    support_count: int,
    strategy: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """为每个旧类按质量策略选择 support recording。"""

    mask = np.zeros(len(y_true), dtype=bool)
    audit: list[dict[str, Any]] = []
    for class_id in range(int(old_end)):
        rows = _recording_table(logits, y_true, recording_id, class_id, old_end)
        selected = _select_recordings(rows, support_count, strategy)
        for row in selected:
            mask[np.asarray(row["sample_idx"], dtype=np.int64)] = True
            audit.append(
                {
                    "class_id": int(class_id),
                    "recording": row["recording"],
                    "true_logit": float(row["true_logit"]),
                    "true_margin": float(row["true_margin"]),
                    "entropy": float(row["entropy"]),
                    "samples": int(len(row["sample_idx"])),
                }
            )
    return mask, audit


def _evaluate_subset(
    stage: str,
    y_true: np.ndarray,
    pred: np.ndarray,
    eval_mask: np.ndarray,
    initial_reference: float | None,
) -> dict[str, float]:
    """剔除 support 样本后复用项目统一指标函数。"""

    return _evaluate(stage, y_true[eval_mask], pred[eval_mask], initial_reference)


def _logreg_pred(
    logits: np.ndarray,
    y_true: np.ndarray,
    base_pred: np.ndarray,
    support_mask: np.ndarray,
    eval_mask: np.ndarray,
    old_end: int,
    c_value: float,
) -> np.ndarray:
    """使用 Stage89 同口径 old-class logreg 校准器预测旧类 eval 样本。"""

    old_eval = eval_mask & (y_true < old_end)
    if not np.any(support_mask):
        return np.asarray(base_pred, dtype=np.int64).copy()
    return _fit_predict_logreg(
        logits[support_mask].astype(np.float32),
        y_true[support_mask].astype(np.int64),
        logits.astype(np.float32),
        base_pred,
        old_eval,
        c_value=float(c_value),
        class_weight=None,
    )


def _collect_strategy(
    save_dir: Path,
    strategy: str,
    support_count: int,
    c_value: float,
) -> dict[str, Any]:
    """收集单个 support 选择策略的逐轮指标和 support audit。"""

    rows: list[dict[str, Any]] = []
    support_audit: dict[str, list[dict[str, Any]]] = {}
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
            audit: list[dict[str, Any]] = []
        else:
            support_mask, audit = _quality_support_mask(
                logits=logits,
                y_true=y_true,
                recording_id=recording_id,
                old_end=old_end,
                support_count=support_count,
                strategy=strategy,
            )
        eval_mask = ~support_mask
        pred = _logreg_pred(logits, y_true, base_pred, support_mask, eval_mask, old_end, c_value)
        metrics = _evaluate_subset(stage_key, y_true, pred, eval_mask, initial_reference)
        if stage_key == "initial":
            initial_reference = float(metrics["initial_known"])
        row: dict[str, Any] = {
            "strategy": strategy,
            "support_recordings_per_old_class": int(support_count),
            "variant": f"{strategy}_logreg_c{str(c_value).replace('.', 'p')}",
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
        support_audit[stage_key] = audit

    r3 = next(row for row in rows if row["stage"] == "After R3")
    return {"strategy": strategy, "rows": rows, "r3": r3, "support_audit": support_audit}


def _fmt(value: Any) -> str:
    """统一四位小数格式。"""

    return f"{float(value):.4f}"


def main() -> int:
    """生成 Stage91 support 质量选择报告。"""

    parser = argparse.ArgumentParser(description="Stage91 LoRa support quality selection")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--support-count", type=int, default=2)
    parser.add_argument("--c-value", type=float, default=0.3)
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    strategies = [item.strip() for item in str(args.strategies).split(",") if item.strip()]
    results = [
        _collect_strategy(
            save_dir=Path(args.save_dir),
            strategy=strategy,
            support_count=int(args.support_count),
            c_value=float(args.c_value),
        )
        for strategy in strategies
    ]
    r3_rows = [item["r3"] for item in results]
    best = max(r3_rows, key=lambda row: float(row["adjusted_original_mix_overall"]))
    pass_count = int(sum(bool(row["passes_adjusted_50"]) for row in r3_rows))
    stable_candidate = pass_count > 0

    lines = [
        f"# Stage91 LoRa support 质量选择诊断（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'至少一种 support 质量选择策略过 50%。' if stable_candidate else '所有 support 质量选择策略仍未过 50%。'}",
        (
            f"- 最佳 R3：`{best['strategy']}`，Adjusted Overall/Old/New="
            f"`{_fmt(best['adjusted_original_mix_overall'])}/{_fmt(best['old'])}/{_fmt(best['new'])}`，"
            f"Old gap to 50=`{_fmt(best['old_gap_to_50'])}`。"
        ),
        f"- 过线策略数：`{pass_count}/{len(r3_rows)}`。",
        "- 边界：这些策略仍从 held-out 旧类真值候选池中选 support；hard 策略还可能因剔除困难样本而高估收益，只能作为诊断。",
        "",
        "## R3 策略对比",
        "",
        "| Strategy | Adjusted Overall | Observed Overall | Old | New | Forgetting | Required Old @50 | Old Gap | Pass | Support Samples | Eval Samples |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(r3_rows, key=lambda item: float(item["adjusted_original_mix_overall"]), reverse=True):
        lines.append(
            "| {strategy} | {adjusted} | {observed} | {old} | {new} | {forgetting} | {required_old} | {old_gap} | {passed} | {support_samples} | {eval_samples} |".format(
                strategy=row["strategy"],
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
                "schema_version": "stage91_lora_support_quality_select_v1",
                "job_id": str(args.job_id),
                "save_dir": str(args.save_dir),
                "support_count": int(args.support_count),
                "c_value": float(args.c_value),
                "strategies": strategies,
                "results": results,
                "best_r3": best,
                "pass_count": pass_count,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Stage91 报告：{output}")
    print(json.dumps({"best_r3": best, "pass_count": pass_count}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
