#!/usr/bin/env python3
"""Stage87：LoRa 同日旧类 support 校准上限诊断。

Stage86 已经证明，在同一个 RADCIL head 里继续叠加旧类跨天监督和新类
margin 只会得到折中，不能真正把 LoRa 推过 Stage46 seed7。因此本阶段不再
改训练，而是回答一个更直接的问题：

如果客户能在每个新增日期额外提供少量“旧设备同日标注 recording”作为
support，剩余 held-out 样本是否能被旧类校准推过 50%？

重要边界：
* 该脚本读取 held-out eval dump 中的 y_true 来抽取 support，所以这是上限/
  数据需求诊断，不是 strict 正式方法成绩。
* support 样本不会计入最终评估分母；它们只用于训练旧类校准器。
* 新类样本保持原主模型预测，避免把未知新类真值用于校准。
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


STAGES = ["initial", "after_r1", "after_r2", "after_r3"]


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    """对 logits 做行归一化，供 prototype 校准器使用。"""

    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def _select_support_mask(
    y_true: np.ndarray,
    recording_id: np.ndarray | None,
    old_end: int,
    recordings_per_old_class: int,
    seed: int,
) -> np.ndarray:
    """按旧类逐类选择 support recording。

    选择粒度优先是完整 recording，而不是随机 symbol。这样更接近真实 LoRa
    场景里“额外标一条 transmission/recording”的成本。若 dump 没有
    recording_id，则退化为每类固定数量样本。
    """

    rng = np.random.default_rng(int(seed))
    y_true = np.asarray(y_true, dtype=np.int64)
    mask = np.zeros(len(y_true), dtype=bool)
    for class_id in range(int(old_end)):
        class_idx = np.flatnonzero(y_true == class_id)
        if len(class_idx) == 0:
            continue
        if recording_id is None:
            take = min(len(class_idx), max(1, int(recordings_per_old_class) * 8))
            mask[np.sort(rng.choice(class_idx, size=take, replace=False))] = True
            continue
        class_recordings = np.unique(np.asarray(recording_id)[class_idx])
        class_recordings = np.asarray(sorted(class_recordings.tolist(), key=lambda item: str(item)))
        take = min(len(class_recordings), max(1, int(recordings_per_old_class)))
        selected = rng.choice(class_recordings, size=take, replace=False)
        mask[class_idx[np.isin(np.asarray(recording_id)[class_idx], selected)]] = True
    return mask


def _prototype_predict(
    logits: np.ndarray,
    y_true: np.ndarray,
    base_pred: np.ndarray,
    support_mask: np.ndarray,
    eval_mask: np.ndarray,
    old_end: int,
) -> np.ndarray:
    """用 support old logits 的类原型重判 eval 中真实旧类，真实新类保持 base。"""

    pred = np.asarray(base_pred, dtype=np.int64).copy()
    logits_norm = _normalise_rows(logits)
    classes: list[int] = []
    prototypes: list[np.ndarray] = []
    for class_id in range(int(old_end)):
        class_support = support_mask & (y_true == class_id)
        if np.any(class_support):
            classes.append(class_id)
            prototypes.append(logits_norm[class_support].mean(axis=0))
    if not prototypes:
        return pred
    proto = _normalise_rows(np.vstack(prototypes))
    old_eval = eval_mask & (y_true < old_end)
    if np.any(old_eval):
        similarity = logits_norm[old_eval] @ proto.T
        nearest = np.argmax(similarity, axis=1)
        pred[old_eval] = np.asarray([classes[int(index)] for index in nearest], dtype=np.int64)
    return pred


def _logreg_predict(
    logits: np.ndarray,
    y_true: np.ndarray,
    base_pred: np.ndarray,
    support_mask: np.ndarray,
    eval_mask: np.ndarray,
    old_end: int,
) -> np.ndarray:
    """用 support old logits 训练线性旧类专家，重判 eval 中真实旧类。"""

    pred = np.asarray(base_pred, dtype=np.int64).copy()
    support_y = np.asarray(y_true[support_mask], dtype=np.int64)
    if len(np.unique(support_y)) < int(old_end):
        return pred
    clf = LogisticRegression(max_iter=3000, solver="lbfgs", n_jobs=1)
    clf.fit(np.asarray(logits[support_mask], dtype=np.float32), support_y)
    old_eval = eval_mask & (y_true < old_end)
    if np.any(old_eval):
        pred[old_eval] = clf.predict(np.asarray(logits[old_eval], dtype=np.float32)).astype(np.int64)
    return pred


def _evaluate_subset(
    stage: str,
    y_true: np.ndarray,
    pred: np.ndarray,
    eval_mask: np.ndarray,
    initial_reference: float | None,
) -> dict[str, float]:
    """复用项目指标函数，但先把 support 样本从评估数组中剔除。"""

    return _evaluate(
        stage,
        np.asarray(y_true[eval_mask], dtype=np.int64),
        np.asarray(pred[eval_mask], dtype=np.int64),
        initial_reference,
    )


def _collect_for_support_count(save_dir: Path, support_count: int, seed: int) -> dict[str, Any]:
    """收集一个 support recording 数量下的逐轮和 R3 指标。"""

    rows: list[dict[str, Any]] = []
    initial_refs: dict[str, float] = {}
    support_stats: list[dict[str, Any]] = []
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
        prototype_pred = _prototype_predict(logits, y_true, base_pred, support_mask, eval_mask, old_end)
        logreg_pred = _logreg_predict(logits, y_true, base_pred, support_mask, eval_mask, old_end)
        variants = {
            "baseline_support_excluded": base_pred,
            "prototype_support_excluded": prototype_pred,
            "logreg_support_excluded": logreg_pred,
        }
        for variant, pred in variants.items():
            ref = None if stage == "initial" else initial_refs.get(variant)
            metrics = _evaluate_subset(stage, y_true, pred, eval_mask, ref)
            if stage == "initial":
                initial_refs[variant] = float(metrics["initial_known"])
            rows.append(
                {
                    "support_recordings_per_old_class": int(support_count),
                    "variant": variant,
                    **metrics,
                    "support_samples": int(np.sum(support_mask)),
                    "eval_samples": int(np.sum(eval_mask)),
                }
            )
        support_stats.append(
            {
                "stage": stage,
                "old_end": old_end,
                "support_samples": int(np.sum(support_mask)),
                "support_old_classes": int(len(np.unique(y_true[support_mask]))) if np.any(support_mask) else 0,
                "eval_samples": int(np.sum(eval_mask)),
                "recording_id_available": recording_id is not None,
            }
        )
    r3_rows = [row for row in rows if row["stage"] == "After R3"]
    selected = max(r3_rows, key=lambda row: (float(row["overall"]), float(row["old"]), float(row["new"])))
    return {
        "support_recordings_per_old_class": int(support_count),
        "rows": rows,
        "r3_rows": r3_rows,
        "selected_r3": selected,
        "support_stats": support_stats,
    }


def _fmt(value: Any) -> str:
    """报告表格统一保留 4 位小数。"""

    return f"{float(value):.4f}"


def main() -> int:
    """生成 Stage87 旧类同日 support 上限诊断报告。"""

    parser = argparse.ArgumentParser(description="Stage87 LoRa same-day old support upper-bound")
    parser.add_argument("--save-dir", required=True, help="包含 end_to_end_eval_dumps 的 Stage68/Stage48 dump 目录")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--support-counts", default="1,2", help="每个旧类抽取多少个 recording 作为 support")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    support_counts = [int(item) for item in str(args.support_counts).split(",") if item.strip()]
    results = [_collect_for_support_count(save_dir, count, args.seed) for count in support_counts]
    best = max((item["selected_r3"] for item in results), key=lambda row: (float(row["overall"]), float(row["old"])))
    passes_50 = float(best["overall"]) >= 0.50

    lines = [
        f"# Stage87 LoRa 同日旧类 support 上限诊断（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'少量同日旧类 support 可把剩余 held-out R3 Overall 推过 50%。' if passes_50 else '少量同日旧类 support 仍未把剩余 held-out R3 Overall 推过 50%。'}",
        f"- 最佳 R3：`{best['variant']}`，support/old-class=`{best['support_recordings_per_old_class']}`，Overall/Old/New/Forgetting=`{_fmt(best['overall'])}/{_fmt(best['old'])}/{_fmt(best['new'])}/{_fmt(best['forgetting'])}`。",
        "- 注意：该实验使用 held-out 旧类真值抽 support，并从评估分母剔除 support；它是数据需求/上限诊断，不是 strict 正式成绩。",
        "",
        "## R3 指标",
        "",
        "| Support / Old Class | Variant | Overall | Old | New | Forgetting | Macro F1 | Support Samples | Eval Samples |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        for row in item["r3_rows"]:
            lines.append(
                "| {support} | {variant} | {overall} | {old} | {new} | {forgetting} | {macro_f1} | {support_samples} | {eval_samples} |".format(
                    support=int(row["support_recordings_per_old_class"]),
                    variant=row["variant"],
                    overall=_fmt(row["overall"]),
                    old=_fmt(row["old"]),
                    new=_fmt(row["new"]),
                    forgetting=_fmt(row["forgetting"]),
                    macro_f1=_fmt(row["macro_f1"]),
                    support_samples=int(row["support_samples"]),
                    eval_samples=int(row["eval_samples"]),
                )
            )
    lines.extend(
        [
            "",
            "## 边界说明",
            "",
            "- support 只来自每轮当日旧类 held-out recording；新类 held-out 不参与校准。",
            "- support 样本不计入最终评估；剩余旧类样本由旧类 prototype/logreg 校准，新类保持主模型预测。",
            "- 如果该上限过 50，说明客户若能提供少量同日旧设备标注，有机会把 LoRa 拉近目标；如果仍不过 50，则仅靠少量旧类 support 也不够。",
            "",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage87_lora_same_day_support_upper_bound_v1",
                "job_id": str(args.job_id),
                "save_dir": str(save_dir),
                "support_counts": support_counts,
                "results": results,
                "best_r3": best,
                "passes_50_overall": bool(passes_50),
                "diagnostic_warning": (
                    "uses held-out old labels to select same-day support; support samples are excluded "
                    "from evaluation; diagnostic upper-bound only"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Stage87 报告：{output}")
    print(json.dumps({"best_r3": best, "passes_50_overall": passes_50}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
