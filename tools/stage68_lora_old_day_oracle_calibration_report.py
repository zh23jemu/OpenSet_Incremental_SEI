#!/usr/bin/env python3
"""Stage68：LoRa 旧类跨天同日校准 oracle 诊断。

这个脚本只读取显式保存的 held-out eval dump，不参与训练、聚类、模型选择或
正式方法调参。它的目的很窄：判断 LoRa 低分是不是主要因为 Day1 旧类缺少
Day2-4 同日校准样本。

诊断口径：
1. baseline：直接使用主模型保存的 pred_mapped。
2. old-day-calibration oracle：对每个旧类，从同一天 held-out eval 中按
   recording_id 取极少量带真值样本作为“假设可获得的同日旧类校准样本”，
   用 logits 均值做旧类原型；评估时只对真实旧类样本启用该校准，新类样本
   保持 baseline 预测。这里显式使用了 held-out 真值，因此只能作为上界诊断，
   不能作为正式方法结果。
3. old-label-remap oracle：在旧类样本上做最乐观的预测标签重映射，用来区分
   “旧类结构还在但标签错位”和“旧类结构本身已经分不开”。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score


STAGES = ("initial", "after_r1", "after_r2", "after_r3")


def _safe_float(value: float) -> float | None:
    """把 NaN 转成 JSON 友好的 None，其它数值保持普通 float。"""

    value = float(value)
    return None if np.isnan(value) else value


def _stage_bounds(stage: str, initial_known: int = 10, round_size: int = 5) -> dict[str, int | None]:
    """返回当前阶段的 seen/old/new 类范围，保持与主实验指标定义一致。"""

    if stage == "initial":
        return {
            "seen_classes": initial_known,
            "old_start": 0,
            "old_end": initial_known,
            "new_start": None,
            "new_end": None,
        }
    round_index = int(stage.replace("after_r", ""))
    old_end = initial_known + (round_index - 1) * round_size
    return {
        "seen_classes": initial_known + round_index * round_size,
        "old_start": 0,
        "old_end": old_end,
        "new_start": old_end,
        "new_end": initial_known + round_index * round_size,
    }


def _acc_on_range(y_true: np.ndarray, y_pred: np.ndarray, start: int, end: int) -> float:
    """计算指定类别区间准确率；没有样本时返回 NaN，便于 Initial 阶段复用。"""

    mask = (y_true >= int(start)) & (y_true < int(end))
    if not np.any(mask):
        return float("nan")
    return float(np.mean(y_true[mask] == y_pred[mask]))


def _evaluate(stage: str, y_true: np.ndarray, y_pred: np.ndarray, initial_reference_acc: float | None) -> dict[str, Any]:
    """按项目主入口一致的 RADCIL 指标口径计算 symbol-level 结果。"""

    bounds = _stage_bounds(stage)
    seen_classes = int(bounds["seen_classes"])
    seen_mask = y_true < seen_classes
    y_seen = y_true[seen_mask]
    pred_seen = y_pred[seen_mask]
    initial_known_acc = _acc_on_range(y_seen, pred_seen, 0, 10)
    if initial_reference_acc is None or np.isnan(initial_known_acc):
        forgetting = 0.0 if stage == "initial" else float("nan")
    else:
        forgetting = float(initial_reference_acc - initial_known_acc)
    new_start = bounds["new_start"]
    new_end = bounds["new_end"]
    new_acc = float("nan") if new_start is None or new_end is None else _acc_on_range(y_seen, pred_seen, int(new_start), int(new_end))
    return {
        "stage": "Initial" if stage == "initial" else f"After R{stage[-1]}",
        "overall": float(np.mean(y_seen == pred_seen)),
        "old": _acc_on_range(y_seen, pred_seen, int(bounds["old_start"]), int(bounds["old_end"])),
        "new": new_acc,
        "initial_known": initial_known_acc,
        "forgetting": forgetting,
        "macro_f1": float(f1_score(y_seen, pred_seen, labels=list(range(seen_classes)), average="macro", zero_division=0)),
    }


def _load_dump(save_dir: Path, stage: str) -> dict[str, np.ndarray]:
    """读取主入口保存的某阶段 eval dump，并校验最少字段。"""

    path = save_dir / "end_to_end_eval_dumps" / f"{stage}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing eval dump: {path}")
    with np.load(path, allow_pickle=False) as data:
        dump = {key: np.asarray(data[key]) for key in data.files}
    required = {"y_true", "pred_mapped", "logits"}
    missing = sorted(required.difference(dump))
    if missing:
        raise KeyError(f"{path} missing required fields: {missing}")
    return dump


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    """做行向量归一化，让 logits 原型匹配主要看方向而不是整体尺度。"""

    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def _select_old_calibration_mask(
    y_true: np.ndarray,
    recording_id: np.ndarray | None,
    old_end: int,
    recordings_per_class: int,
    samples_per_class_fallback: int,
) -> np.ndarray:
    """为每个旧类选择少量“同日校准”样本。

    优先按 recording_id 选择完整 recording，模拟实际采集中按 transmission/recording
    拿少量标注校准样本；如果 dump 没有 recording_id，则退化为每类前若干个样本。
    该选择只用于 oracle 诊断，显式依赖 held-out y_true。
    """

    y_true = np.asarray(y_true, dtype=np.int64)
    mask = np.zeros_like(y_true, dtype=bool)
    for class_id in range(int(old_end)):
        class_idx = np.flatnonzero(y_true == class_id)
        if len(class_idx) == 0:
            continue
        if recording_id is None:
            mask[class_idx[: max(1, int(samples_per_class_fallback))]] = True
            continue
        class_groups = np.asarray(recording_id)[class_idx]
        unique_groups = sorted(np.unique(class_groups).tolist(), key=lambda item: str(item))
        selected_groups = set(unique_groups[: max(1, int(recordings_per_class))])
        mask[class_idx[np.isin(class_groups, list(selected_groups))]] = True
    return mask


def _old_day_calibration_oracle(
    dump: dict[str, np.ndarray],
    stage: str,
    recordings_per_class: int,
    samples_per_class_fallback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """用同日旧类校准样本建立 logits 原型，并只在真实旧类样本上应用。

    返回值包含 oracle 预测和被当作校准样本的布尔 mask。评估时新类预测不动，
    旧类非校准样本通过最近旧类原型重判，校准样本按已知标签记为正确。
    """

    y_true = dump["y_true"].astype(np.int64)
    pred = dump["pred_mapped"].astype(np.int64).copy()
    logits = _normalise_rows(dump["logits"])
    bounds = _stage_bounds(stage)
    old_end = int(bounds["old_end"])
    recording_id = dump.get("recording_id")
    calibration_mask = _select_old_calibration_mask(
        y_true,
        recording_id,
        old_end=old_end,
        recordings_per_class=recordings_per_class,
        samples_per_class_fallback=samples_per_class_fallback,
    )

    prototype_classes: list[int] = []
    prototypes: list[np.ndarray] = []
    for class_id in range(old_end):
        class_mask = calibration_mask & (y_true == class_id)
        if not np.any(class_mask):
            continue
        prototype_classes.append(class_id)
        prototypes.append(logits[class_mask].mean(axis=0))
    if not prototypes:
        return pred, calibration_mask

    proto = _normalise_rows(np.vstack(prototypes))
    old_eval_mask = y_true < old_end
    old_test_mask = old_eval_mask & ~calibration_mask
    if np.any(old_test_mask):
        similarity = logits[old_test_mask] @ proto.T
        nearest = np.argmax(similarity, axis=1)
        pred[old_test_mask] = np.asarray([prototype_classes[int(idx)] for idx in nearest], dtype=np.int64)
    pred[calibration_mask & old_eval_mask] = y_true[calibration_mask & old_eval_mask]
    return pred, calibration_mask


def _old_label_remap_oracle(dump: dict[str, np.ndarray], stage: str) -> np.ndarray:
    """旧类标签重映射 oracle，用来判断旧类错误是否只是类编号错位。

    它在真实旧类样本上用 Hungarian 匹配最大化 pred_mapped 与 y_true 的对应关系。
    这是更强的上界，使用了全部旧类评估真值，绝不能作为正式方法。
    """

    y_true = dump["y_true"].astype(np.int64)
    pred = dump["pred_mapped"].astype(np.int64).copy()
    old_end = int(_stage_bounds(stage)["old_end"])
    old_mask = y_true < old_end
    if not np.any(old_mask):
        return pred
    true_labels = list(range(old_end))
    pred_labels = sorted(np.unique(pred[old_mask]).tolist())
    if not pred_labels:
        return pred
    matrix = np.zeros((len(true_labels), len(pred_labels)), dtype=np.int64)
    pred_index = {int(label): idx for idx, label in enumerate(pred_labels)}
    for true_value, pred_value in zip(y_true[old_mask], pred[old_mask]):
        matrix[int(true_value), pred_index[int(pred_value)]] += 1
    row_ind, col_ind = linear_sum_assignment(-matrix)
    mapping = {int(pred_labels[col]): int(true_labels[row]) for row, col in zip(row_ind, col_ind)}
    remapped_old = np.asarray([mapping.get(int(value), int(value)) for value in pred[old_mask]], dtype=np.int64)
    pred[old_mask] = remapped_old
    return pred


def _fmt(value: float | None) -> str:
    """统一报告里的小数格式，None/NaN 显示为短横线。"""

    if value is None:
        return "-"
    return f"{float(value):.4f}"


def collect_records(save_dir: Path, recordings_per_class: int, samples_per_class_fallback: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """收集 baseline、同日校准 oracle 和旧标签重映射 oracle 的逐阶段指标。"""

    records: list[dict[str, Any]] = []
    initial_refs: dict[str, float] = {}
    diagnostics: dict[str, Any] = {"save_dir": str(save_dir), "calibration": []}
    for stage in STAGES:
        dump = _load_dump(save_dir, stage)
        y_true = dump["y_true"].astype(np.int64)
        baseline_pred = dump["pred_mapped"].astype(np.int64)
        calibration_pred, calibration_mask = _old_day_calibration_oracle(
            dump,
            stage,
            recordings_per_class=recordings_per_class,
            samples_per_class_fallback=samples_per_class_fallback,
        )
        remap_pred = _old_label_remap_oracle(dump, stage)
        variants = {
            "baseline": baseline_pred,
            "old_day_calibration_oracle": calibration_pred,
            "old_label_remap_oracle": remap_pred,
        }
        for variant, pred in variants.items():
            ref = None if stage == "initial" else initial_refs.get(variant)
            metrics = _evaluate(stage, y_true, pred, ref)
            if stage == "initial":
                initial_refs[variant] = float(metrics["initial_known"])
            records.append({"variant": variant, **{key: _safe_float(value) if isinstance(value, float) else value for key, value in metrics.items()}})
        old_end = int(_stage_bounds(stage)["old_end"])
        diagnostics["calibration"].append(
            {
                "stage": stage,
                "old_end": old_end,
                "calibration_samples": int(np.sum(calibration_mask & (y_true < old_end))),
                "old_eval_samples": int(np.sum(y_true < old_end)),
                "recording_id_available": "recording_id" in dump,
            }
        )
    return records, diagnostics


def _r3(records: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    """取指定变体的 R3 行。"""

    rows = [row for row in records if row["variant"] == variant and row["stage"] == "After R3"]
    if not rows:
        raise ValueError(f"Missing R3 row for {variant}")
    return rows[-1]


def _table(records: list[dict[str, Any]]) -> str:
    """生成 R3 主表，便于直接粘进报告或客户口径。"""

    lines = [
        "| Variant | Overall | Old | New | Forgetting | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in ("baseline", "old_day_calibration_oracle", "old_label_remap_oracle"):
        row = _r3(records, variant)
        lines.append(
            "| {variant} | {overall} | {old} | {new} | {forgetting} | {macro_f1} |".format(
                variant=variant,
                overall=_fmt(row["overall"]),
                old=_fmt(row["old"]),
                new=_fmt(row["new"]),
                forgetting=_fmt(row["forgetting"]),
                macro_f1=_fmt(row["macro_f1"]),
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage68 LoRa old-day oracle calibration report")
    parser.add_argument("--save-dir", required=True, help="包含 end_to_end_eval_dumps 的 Stage68 复跑目录")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--recordings-per-class", type=int, default=1)
    parser.add_argument("--samples-per-class-fallback", type=int, default=8)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    records, diagnostics = collect_records(
        save_dir,
        recordings_per_class=args.recordings_per_class,
        samples_per_class_fallback=args.samples_per_class_fallback,
    )
    baseline = _r3(records, "baseline")
    calibrated = _r3(records, "old_day_calibration_oracle")
    remapped = _r3(records, "old_label_remap_oracle")
    reaches_50 = float(calibrated["overall"]) >= 0.50
    old_gap_closed = float(calibrated["old"]) >= 0.50

    lines = [
        f"# Stage68 LoRa 旧类跨天同日校准 Oracle 诊断（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'同日旧类校准上界已接近/超过 50%，说明主要瓶颈是缺跨天旧类校准数据。' if reaches_50 else '即使用同日旧类 oracle 校准，R3 Overall 仍未到 50%，说明瓶颈不只是缺少少量旧类校准样本。'}",
        f"- Baseline R3 Overall/Old/New = {_fmt(baseline['overall'])}/{_fmt(baseline['old'])}/{_fmt(baseline['new'])}。",
        f"- 同日旧类校准 oracle R3 Overall/Old/New = {_fmt(calibrated['overall'])}/{_fmt(calibrated['old'])}/{_fmt(calibrated['new'])}，Old 是否达到 0.50：{'是' if old_gap_closed else '否'}。",
        f"- 旧类标签重映射 oracle R3 Overall/Old/New = {_fmt(remapped['overall'])}/{_fmt(remapped['old'])}/{_fmt(remapped['new'])}，用于观察旧类是否只是类编号错位。",
        "- 注意：两个 oracle 都读取 held-out 真值，只能解释瓶颈，不能作为正式方法或论文主结果。",
        "",
        "## R3 指标",
        "",
        _table(records),
        "",
        "## 诊断设置",
        "",
        f"- 复跑目录：`{save_dir}`",
        f"- 每个旧类最多取 `{args.recordings_per_class}` 个 recording 做同日校准；没有 recording_id 时每类取 `{args.samples_per_class_fallback}` 个样本。",
        "- 新类样本保持 baseline 预测；old-day-calibration oracle 只在真实旧类样本上应用，因此是旧类跨天校准上界。",
        "",
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage68_lora_old_day_oracle_calibration_v1",
                "job_id": str(args.job_id),
                "save_dir": str(save_dir),
                "records": records,
                "diagnostics": diagnostics,
                "r3": {
                    "baseline": baseline,
                    "old_day_calibration_oracle": calibrated,
                    "old_label_remap_oracle": remapped,
                },
                "gate": {
                    "old_day_calibration_overall_reaches_50": reaches_50,
                    "old_day_calibration_old_reaches_50": old_gap_closed,
                },
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
