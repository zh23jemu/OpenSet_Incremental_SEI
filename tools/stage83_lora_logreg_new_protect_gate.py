"""Stage83：LoRa 旧类 logreg 校准 + 新类保护门控。

Stage73 的旧类 logreg 校准能把 R3 Overall 推到接近 50%，但它使用
held-out eval 真值中的“旧类 mask”来决定哪些样本可被旧类专家改写，因此
只能作为诊断上限，不能直接当作可部署方法。本阶段把这个 oracle mask 换成
训练期可得到的来源门控：旧类专家只在“旧类来源概率足够高，且主模型没有
高置信新类证据”时生效。

严格边界：
* 旧类专家只使用 Day2-4 旧设备 IQ_1-7 标注校准样本训练；
* 来源门控只使用旧类校准样本和当前轮 discovery/enrollment 样本选阈值；
* held-out IQ_8-10 标签只在最后计算报告指标时读取，不参与阈值和路由选择。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.lora25_strict_loader import load_lora25_diffdays_3round  # noqa: E402
from tools.stage69_lora_old_day_calibration_report import (  # noqa: E402
    _build_calibration_x,
    _evaluate,
    _load_lora_stage_model,
    _stage_bounds,
)
from tools.stage77_lora_old_new_expert_gate import _checkpoint_for_round, _split_indices  # noqa: E402
from tools.stage78_lora_source_aware_gate import (  # noqa: E402
    _extract_outputs,
    _old_prototypes,
    _source_features,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """稳定计算 softmax，用于构造主模型置信度和 margin。"""

    values = np.asarray(logits, dtype=np.float32)
    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-8)


def _main_confidence_features(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回主模型预测、top-1 概率和 top-1/top-2 概率间隔。"""

    probs = _softmax(logits)
    main_pred = probs.argmax(axis=1).astype(np.int64)
    top2 = np.partition(probs, kth=max(0, probs.shape[1] - 2), axis=1)[:, -2:]
    confidence = top2[:, 1]
    margin = top2[:, 1] - top2[:, 0]
    return main_pred, confidence.astype(np.float32), margin.astype(np.float32)


def _fit_old_source_gate_with_new_protection(
    old_logits: np.ndarray,
    old_features: np.ndarray,
    discovery_logits: np.ndarray,
    discovery_features: np.ndarray,
    prototypes: np.ndarray,
    old_end: int,
    seed: int,
) -> tuple[LogisticRegression, dict[str, float]]:
    """训练旧/新来源门控，并只用训练期样本选择新类保护阈值。

    阈值搜索的目标不是最大化 held-out 准确率，而是在旧类校准留出样本与
    当前 discovery 留出样本之间取得平衡：旧类样本要尽量被旧专家接住，
    discovery 样本要尽量不被误吸回旧类。这样可以模拟真实部署时“旧类救回”
    与“新类保护”的权衡。
    """

    old_x = _source_features(old_logits, old_features, prototypes)
    discovery_x = _source_features(discovery_logits, discovery_features, prototypes)
    old_train, old_valid = _split_indices(len(old_x), seed)
    discovery_train, discovery_valid = _split_indices(len(discovery_x), seed + 1)

    train_x = np.concatenate([old_x[old_train], discovery_x[discovery_train]], axis=0)
    train_y = np.concatenate(
        [np.ones(len(old_train), dtype=np.int64), np.zeros(len(discovery_train), dtype=np.int64)],
        axis=0,
    )
    gate = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed, n_jobs=1)
    gate.fit(train_x, train_y)
    old_index = int(np.where(gate.classes_ == 1)[0][0])

    valid_logits = np.concatenate([old_logits[old_valid], discovery_logits[discovery_valid]], axis=0)
    valid_x = np.concatenate([old_x[old_valid], discovery_x[discovery_valid]], axis=0)
    valid_y = np.concatenate(
        [np.ones(len(old_valid), dtype=np.int64), np.zeros(len(discovery_valid), dtype=np.int64)],
        axis=0,
    )
    valid_old_prob = gate.predict_proba(valid_x)[:, old_index]
    valid_main_pred, valid_main_conf, valid_main_margin = _main_confidence_features(valid_logits)
    valid_main_new = valid_main_pred >= int(old_end)

    # 新类保护阈值只从 discovery 留出分布中取分位数，避免读取 eval 真值。
    discovery_pred, discovery_conf, discovery_margin = _main_confidence_features(discovery_logits[discovery_valid])
    discovery_new_mask = discovery_pred >= int(old_end)
    if np.any(discovery_new_mask):
        conf_candidates = np.unique(np.quantile(discovery_conf[discovery_new_mask], [0.25, 0.50, 0.75])).tolist()
        margin_candidates = np.unique(np.quantile(discovery_margin[discovery_new_mask], [0.25, 0.50, 0.75])).tolist()
    else:
        conf_candidates = [0.0]
        margin_candidates = [0.0]

    scored: list[tuple[float, float, float, float, float, float, float, float]] = []
    for gate_threshold in np.linspace(0.10, 0.95, 18):
        for conf_threshold in conf_candidates:
            for margin_threshold in margin_candidates:
                protected_new = valid_main_new & (
                    (valid_main_conf >= float(conf_threshold)) | (valid_main_margin >= float(margin_threshold))
                )
                route_old = (valid_old_prob >= float(gate_threshold)) & ~protected_new
                old_recall = float(np.mean(route_old[valid_y == 1])) if np.any(valid_y == 1) else 0.0
                discovery_reject = float(np.mean(~route_old[valid_y == 0])) if np.any(valid_y == 0) else 0.0
                route_fraction = float(np.mean(route_old))
                # 先保证旧/新来源平衡，再偏向更保守的 discovery 保护，最后偏向少路由。
                score = min(old_recall, discovery_reject)
                tie = discovery_reject + 0.5 * old_recall - 0.05 * route_fraction
                scored.append(
                    (
                        score,
                        tie,
                        float(gate_threshold),
                        float(conf_threshold),
                        float(margin_threshold),
                        old_recall,
                        discovery_reject,
                        route_fraction,
                    )
                )

    _, _, gate_threshold, conf_threshold, margin_threshold, old_recall, discovery_reject, route_fraction = max(scored)
    full_x = np.concatenate([old_x, discovery_x], axis=0)
    full_y = np.concatenate([np.ones(len(old_x), dtype=np.int64), np.zeros(len(discovery_x), dtype=np.int64)], axis=0)
    gate.fit(full_x, full_y)
    return gate, {
        "gate_threshold": gate_threshold,
        "new_conf_threshold": conf_threshold,
        "new_margin_threshold": margin_threshold,
        "valid_old_source_recall": old_recall,
        "valid_discovery_reject": discovery_reject,
        "valid_route_fraction": route_fraction,
    }


def _predict_with_new_protected_old_logreg(
    eval_logits: np.ndarray,
    eval_features: np.ndarray,
    old_logits: np.ndarray,
    old_labels: np.ndarray,
    prototypes: np.ndarray,
    gate: LogisticRegression,
    gate_info: dict[str, float],
    old_end: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """应用 Stage83 后处理，并返回路由诊断信息。

    旧类专家使用 logits-space logreg，继承 Stage73 的有效信号；新类保护由主模型
    top-1 是否落在当前 seen 新类范围、以及 top-1 置信度/margin 是否超过训练期
    discovery 分布阈值共同决定。
    """

    logits = np.asarray(eval_logits, dtype=np.float32)
    old_expert = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0, n_jobs=1)
    old_expert.fit(np.asarray(old_logits, dtype=np.float32), np.asarray(old_labels, dtype=np.int64))

    main_pred, main_conf, main_margin = _main_confidence_features(logits)
    old_pred = old_expert.predict(logits).astype(np.int64)
    old_index = int(np.where(gate.classes_ == 1)[0][0])
    gate_x = _source_features(logits, eval_features, prototypes)
    old_probability = gate.predict_proba(gate_x)[:, old_index]

    main_new = main_pred >= int(old_end)
    protected_new = main_new & (
        (main_conf >= float(gate_info["new_conf_threshold"]))
        | (main_margin >= float(gate_info["new_margin_threshold"]))
    )
    route_old = (old_probability >= float(gate_info["gate_threshold"])) & ~protected_new
    corrected = main_pred.copy()
    corrected[route_old] = old_pred[route_old]

    return corrected, {
        "old_route_fraction": float(route_old.mean()),
        "protected_new_fraction": float(protected_new.mean()),
        "main_new_fraction": float(main_new.mean()),
        "mean_old_probability": float(old_probability.mean()),
        "mean_main_confidence": float(main_conf.mean()),
        "mean_main_margin": float(main_margin.mean()),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出逐轮指标，便于和 Stage73/77/79 横向对齐。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["Round"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage83 LoRa 旧类 logreg 校准 + 新类保护门控")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--stage-save-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--transmissions", default="1,2,3,4,5,6,7")
    parser.add_argument("--symbols-per-transmission", type=int, default=28)
    parser.add_argument("--decimation", type=int, default=1)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(args.stage_save_dir).resolve()
    transmissions = [int(value) for value in args.transmissions.split(",") if value.strip()]
    protocol = load_lora25_diffdays_3round(Path(args.dataset_path))

    rows: list[dict[str, Any]] = []
    main_initial_reference = None
    stage83_initial_reference = None

    for round_index in range(1, 4):
        stage = f"after_r{round_index}"
        bounds = _stage_bounds(stage)
        seen_classes = int(bounds["seen_classes"])
        old_end = int(bounds["old_end"])
        model = _load_lora_stage_model(
            _checkpoint_for_round(stage_dir, round_index),
            seen_classes=seen_classes,
            feat_dim=args.feat_dim,
            device=device,
        )

        calibration_x, calibration_y, _ = _build_calibration_x(
            Path(args.raw_dir),
            day=round_index + 1,
            old_end=old_end,
            transmissions=transmissions,
            symbols_per_transmission=args.symbols_per_transmission,
            decimation=args.decimation,
            representation="raw",
        )
        calibration_features, calibration_logits = _extract_outputs(model, calibration_x, args.batch_size, device)

        discovery_key = f"day{round_index + 1}_unknown_round{round_index}"
        discovery_features, discovery_logits = _extract_outputs(
            model,
            np.asarray(protocol[discovery_key]["X"]),
            args.batch_size,
            device,
        )
        prototypes = _old_prototypes(calibration_features, calibration_y, old_end)
        gate, gate_info = _fit_old_source_gate_with_new_protection(
            old_logits=calibration_logits,
            old_features=calibration_features,
            discovery_logits=discovery_logits,
            discovery_features=discovery_features,
            prototypes=prototypes,
            old_end=old_end,
            seed=args.seed + round_index,
        )

        eval_key = f"day{round_index + 1}_eval_after_r{round_index}"
        eval_stage = protocol[eval_key]
        eval_features, eval_logits = _extract_outputs(model, np.asarray(eval_stage["X"]), args.batch_size, device)
        y_true = np.asarray(eval_stage["y"])

        main_prediction = np.asarray(eval_logits, dtype=np.float32).argmax(axis=1).astype(np.int64)
        stage83_prediction, route_info = _predict_with_new_protected_old_logreg(
            eval_logits=eval_logits,
            eval_features=eval_features,
            old_logits=calibration_logits,
            old_labels=calibration_y,
            prototypes=prototypes,
            gate=gate,
            gate_info=gate_info,
            old_end=old_end,
        )

        main_metrics = _evaluate(stage, y_true, main_prediction, main_initial_reference)
        if main_initial_reference is None:
            main_initial_reference = float(main_metrics["initial_known"])
        metrics = _evaluate(stage, y_true, stage83_prediction, stage83_initial_reference)
        if stage83_initial_reference is None:
            stage83_initial_reference = float(metrics["initial_known"])

        rows.append(
            {
                "Round": f"R{round_index}",
                "Overall Acc": metrics["overall"],
                "Old Acc": metrics["old"],
                "New Acc": metrics["new"],
                "Forgetting Rate": metrics["forgetting"],
                "Macro F1": metrics["macro_f1"],
                "Main Overall Acc": main_metrics["overall"],
                "Main Old Acc": main_metrics["old"],
                "Main New Acc": main_metrics["new"],
                "Main Forgetting Rate": main_metrics["forgetting"],
                "Old Route Fraction": route_info["old_route_fraction"],
                "Protected New Fraction": route_info["protected_new_fraction"],
                "Main New Prediction Fraction": route_info["main_new_fraction"],
                "Mean Old Probability": route_info["mean_old_probability"],
                "Gate Threshold": gate_info["gate_threshold"],
                "New Confidence Threshold": gate_info["new_conf_threshold"],
                "New Margin Threshold": gate_info["new_margin_threshold"],
                "Valid Old Source Recall": gate_info["valid_old_source_recall"],
                "Valid Discovery Reject": gate_info["valid_discovery_reject"],
                "Valid Route Fraction": gate_info["valid_route_fraction"],
            }
        )

    _write_csv(output_dir / "stage83_lora_logreg_new_protect_gate_results.csv", rows)
    r3 = rows[-1]
    reference = {"overall": 0.4695, "old": 0.4488, "new": 0.5524, "forgetting": 0.0774}
    deployable_reference = {"overall": 0.2810, "old": 0.2220, "new": 0.5167, "forgetting": 0.1726}
    passes = (
        r3["Overall Acc"] >= deployable_reference["overall"] + 0.02
        and r3["Old Acc"] >= deployable_reference["old"] + 0.02
        and r3["New Acc"] >= deployable_reference["new"] - 0.05
    )
    summary = {
        "schema_version": "stage83_lora_logreg_new_protect_gate_v1",
        "method": "old_logreg_calibration_with_new_protection_gate",
        "protocol": (
            "old calibration IQ_1-7 + current discovery source gate; threshold selection excludes "
            "held-out IQ_8-10 truth"
        ),
        "stage_save_dir": str(stage_dir),
        "device": device,
        "rows": rows,
        "r3": r3,
        "diagnostic_upper_bound_stage73_iq1_7": reference,
        "deployable_stage48_seed7_reference": deployable_reference,
        "delta_vs_stage73_oracle_old_mask": {
            "overall": r3["Overall Acc"] - reference["overall"],
            "old": r3["Old Acc"] - reference["old"],
            "new": r3["New Acc"] - reference["new"],
            "forgetting": r3["Forgetting Rate"] - reference["forgetting"],
        },
        "delta_vs_stage48_seed7": {
            "overall": r3["Overall Acc"] - deployable_reference["overall"],
            "old": r3["Old Acc"] - deployable_reference["old"],
            "new": r3["New Acc"] - deployable_reference["new"],
            "forgetting": r3["Forgetting Rate"] - deployable_reference["forgetting"],
        },
        "gate": passes,
        "decision": (
            "pass_seed7_candidate" if passes else "negative_or_diagnostic_only_do_not_expand"
        ),
    }
    (output_dir / "stage83_lora_logreg_new_protect_gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "STAGE83_LORA_LOGREG_NEW_PROTECT_GATE_SEED7_REPORT.md").write_text(
        "# Stage83 LoRa 旧类 logreg 校准 + 新类保护门控 seed7 报告\n\n"
        f"- R3 Overall/Old/New/Forgetting: `{r3['Overall Acc']:.4f}/"
        f"{r3['Old Acc']:.4f}/{r3['New Acc']:.4f}/{r3['Forgetting Rate']:.4f}`\n"
        f"- 主模型原始 R3 Overall/Old/New: `{r3['Main Overall Acc']:.4f}/"
        f"{r3['Main Old Acc']:.4f}/{r3['Main New Acc']:.4f}`\n"
        f"- Stage73 oracle old-mask 诊断上限 R3 Overall/Old/New: "
        f"`{reference['overall']:.4f}/{reference['old']:.4f}/{reference['new']:.4f}`\n"
        f"- 旧类路由比例：`{r3['Old Route Fraction']:.4f}`；"
        f"新类保护比例：`{r3['Protected New Fraction']:.4f}`；"
        f"门控阈值：`{r3['Gate Threshold']:.2f}`。\n"
        f"- 通过门槛：`{passes}`；决策：`{summary['decision']}`。\n"
        "- 说明：本阶段不使用 held-out eval 真值做路由，目标是把 Stage73 的诊断上限转成可部署门控。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
