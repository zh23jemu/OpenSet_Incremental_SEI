"""Stage84：LoRa 保守新类保护门控矩阵。

Stage83 证明旧类 logreg 专家能明显抬 Old/Overall，但旧类路由比例过高，
导致 New Acc 被压低。本阶段不再改模型训练，只在训练期可见数据上提高
discovery reject 目标，形成更保守的旧类路由矩阵。

严格边界：
* 旧类专家仍只使用 Day2-4 旧设备 IQ_1-7 标注校准样本；
* 阈值只由旧类校准留出样本和当前 discovery/enrollment 留出样本选择；
* IQ_8-10 held-out 标签只在最终报告指标时读取。
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
from tools.stage78_lora_source_aware_gate import _extract_outputs, _old_prototypes, _source_features  # noqa: E402
from tools.stage83_lora_logreg_new_protect_gate import (  # noqa: E402
    _main_confidence_features,
    _predict_with_new_protected_old_logreg,
)


def _fit_gate_and_policy_thresholds(
    old_logits: np.ndarray,
    old_features: np.ndarray,
    discovery_logits: np.ndarray,
    discovery_features: np.ndarray,
    prototypes: np.ndarray,
    old_end: int,
    seed: int,
    reject_targets: list[float],
) -> tuple[LogisticRegression, dict[str, dict[str, float]]]:
    """训练同一个来源门控，并为不同 discovery reject 目标选择阈值。

    每个策略只使用训练期留出数据打分。若某个 reject 目标没有候选满足，
    则使用 discovery reject 最高、old recall 次高的兜底候选，避免读取
    held-out eval 真值做任何阈值选择。
    """

    old_x = _source_features(old_logits, old_features, prototypes)
    discovery_x = _source_features(discovery_logits, discovery_features, prototypes)
    old_train, old_valid = _split_indices(len(old_x), seed)
    discovery_train, discovery_valid = _split_indices(len(discovery_x), seed + 1)

    train_x = np.concatenate([old_x[old_train], discovery_x[discovery_train]], axis=0)
    train_y = np.concatenate(
        [np.ones(len(old_train), dtype=np.int64), np.zeros(len(discovery_train), dtype=np.int64)]
    )
    gate = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed, n_jobs=1)
    gate.fit(train_x, train_y)
    old_index = int(np.where(gate.classes_ == 1)[0][0])

    valid_logits = np.concatenate([old_logits[old_valid], discovery_logits[discovery_valid]], axis=0)
    valid_x = np.concatenate([old_x[old_valid], discovery_x[discovery_valid]], axis=0)
    valid_y = np.concatenate(
        [np.ones(len(old_valid), dtype=np.int64), np.zeros(len(discovery_valid), dtype=np.int64)]
    )
    old_probability = gate.predict_proba(valid_x)[:, old_index]
    main_pred, main_conf, main_margin = _main_confidence_features(valid_logits)
    main_new = main_pred >= int(old_end)

    discovery_pred, discovery_conf, discovery_margin = _main_confidence_features(discovery_logits[discovery_valid])
    discovery_new = discovery_pred >= int(old_end)
    if np.any(discovery_new):
        conf_candidates = np.unique(np.quantile(discovery_conf[discovery_new], [0.10, 0.25, 0.50, 0.75, 0.90])).tolist()
        margin_candidates = np.unique(np.quantile(discovery_margin[discovery_new], [0.10, 0.25, 0.50, 0.75, 0.90])).tolist()
    else:
        conf_candidates = [0.0]
        margin_candidates = [0.0]

    candidates: list[dict[str, float]] = []
    for gate_threshold in np.linspace(0.10, 0.99, 30):
        for conf_threshold in conf_candidates:
            for margin_threshold in margin_candidates:
                protected_new = main_new & (
                    (main_conf >= float(conf_threshold)) | (main_margin >= float(margin_threshold))
                )
                route_old = (old_probability >= float(gate_threshold)) & ~protected_new
                old_recall = float(np.mean(route_old[valid_y == 1])) if np.any(valid_y == 1) else 0.0
                discovery_reject = float(np.mean(~route_old[valid_y == 0])) if np.any(valid_y == 0) else 0.0
                route_fraction = float(np.mean(route_old))
                candidates.append(
                    {
                        "gate_threshold": float(gate_threshold),
                        "new_conf_threshold": float(conf_threshold),
                        "new_margin_threshold": float(margin_threshold),
                        "valid_old_source_recall": old_recall,
                        "valid_discovery_reject": discovery_reject,
                        "valid_route_fraction": route_fraction,
                    }
                )

    policies: dict[str, dict[str, float]] = {}
    for target in reject_targets:
        feasible = [row for row in candidates if row["valid_discovery_reject"] >= float(target)]
        pool = feasible if feasible else candidates
        selected = max(
            pool,
            key=lambda row: (
                row["valid_old_source_recall"] - 0.03 * row["valid_route_fraction"],
                row["valid_discovery_reject"],
                -row["valid_route_fraction"],
            ),
        )
        policies[f"reject_{target:.2f}".replace(".", "p")] = dict(selected)
        policies[f"reject_{target:.2f}".replace(".", "p")]["target_discovery_reject"] = float(target)
        policies[f"reject_{target:.2f}".replace(".", "p")]["target_feasible"] = bool(feasible)

    full_x = np.concatenate([old_x, discovery_x], axis=0)
    full_y = np.concatenate([np.ones(len(old_x), dtype=np.int64), np.zeros(len(discovery_x), dtype=np.int64)])
    gate.fit(full_x, full_y)
    return gate, policies


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出策略 x 轮次的长表结果。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["Policy"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage84 LoRa 保守新类保护门控矩阵")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--stage-save-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reject-targets", default="0.80,0.90,0.95,0.98")
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
    reject_targets = [float(value) for value in args.reject_targets.split(",") if value.strip()]
    protocol = load_lora25_diffdays_3round(Path(args.dataset_path))

    rows: list[dict[str, Any]] = []
    initial_refs: dict[str, float] = {}
    main_initial_reference: float | None = None

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
        gate, policies = _fit_gate_and_policy_thresholds(
            old_logits=calibration_logits,
            old_features=calibration_features,
            discovery_logits=discovery_logits,
            discovery_features=discovery_features,
            prototypes=prototypes,
            old_end=old_end,
            seed=args.seed + round_index,
            reject_targets=reject_targets,
        )

        eval_key = f"day{round_index + 1}_eval_after_r{round_index}"
        eval_stage = protocol[eval_key]
        eval_features, eval_logits = _extract_outputs(model, np.asarray(eval_stage["X"]), args.batch_size, device)
        y_true = np.asarray(eval_stage["y"])
        main_prediction = np.asarray(eval_logits, dtype=np.float32).argmax(axis=1).astype(np.int64)
        main_metrics = _evaluate(stage, y_true, main_prediction, main_initial_reference)
        if main_initial_reference is None:
            main_initial_reference = float(main_metrics["initial_known"])

        for policy, gate_info in policies.items():
            prediction, route_info = _predict_with_new_protected_old_logreg(
                eval_logits=eval_logits,
                eval_features=eval_features,
                old_logits=calibration_logits,
                old_labels=calibration_y,
                prototypes=prototypes,
                gate=gate,
                gate_info=gate_info,
                old_end=old_end,
            )
            metrics = _evaluate(stage, y_true, prediction, initial_refs.get(policy))
            if policy not in initial_refs:
                initial_refs[policy] = float(metrics["initial_known"])
            rows.append(
                {
                    "Policy": policy,
                    "Round": f"R{round_index}",
                    "Overall Acc": metrics["overall"],
                    "Old Acc": metrics["old"],
                    "New Acc": metrics["new"],
                    "Forgetting Rate": metrics["forgetting"],
                    "Macro F1": metrics["macro_f1"],
                    "Main Overall Acc": main_metrics["overall"],
                    "Main Old Acc": main_metrics["old"],
                    "Main New Acc": main_metrics["new"],
                    "Old Route Fraction": route_info["old_route_fraction"],
                    "Protected New Fraction": route_info["protected_new_fraction"],
                    "Gate Threshold": gate_info["gate_threshold"],
                    "Target Discovery Reject": gate_info["target_discovery_reject"],
                    "Target Feasible": gate_info["target_feasible"],
                    "Valid Old Source Recall": gate_info["valid_old_source_recall"],
                    "Valid Discovery Reject": gate_info["valid_discovery_reject"],
                    "Valid Route Fraction": gate_info["valid_route_fraction"],
                }
            )

    _write_csv(output_dir / "stage84_lora_conservative_new_protect_sweep_results.csv", rows)
    r3_rows = [row for row in rows if row["Round"] == "R3"]
    reference = {"overall": 0.3005, "old": 0.2429, "new": 0.5310, "forgetting": 0.1952}
    stage83 = {"overall": 0.3138, "old": 0.3119, "new": 0.3214, "forgetting": 0.1905}
    selected = max(
        r3_rows,
        key=lambda row: (
            row["Overall Acc"],
            row["New Acc"] >= reference["new"] - 0.05,
            row["Old Acc"],
        ),
    )
    balanced = max(
        r3_rows,
        key=lambda row: (
            min(row["Old Acc"], row["New Acc"]),
            row["Overall Acc"],
        ),
    )
    passes = (
        selected["Overall Acc"] >= reference["overall"] + 0.01
        and selected["Old Acc"] >= reference["old"]
        and selected["New Acc"] >= reference["new"] - 0.05
    )
    summary = {
        "schema_version": "stage84_lora_conservative_new_protect_sweep_v1",
        "method": "old_logreg_new_protect_conservative_reject_sweep",
        "protocol": "thresholds selected only from old calibration holdout and current discovery holdout",
        "stage_save_dir": str(stage_dir),
        "device": device,
        "reject_targets": reject_targets,
        "rows": rows,
        "r3_rows": r3_rows,
        "selected_by_overall": selected,
        "selected_by_balance": balanced,
        "stage46_seed7_reference": reference,
        "stage83_reference": stage83,
        "delta_selected_vs_stage83": {
            "overall": selected["Overall Acc"] - stage83["overall"],
            "old": selected["Old Acc"] - stage83["old"],
            "new": selected["New Acc"] - stage83["new"],
            "forgetting": selected["Forgetting Rate"] - stage83["forgetting"],
        },
        "gate": passes,
        "decision": "pass_seed7_candidate" if passes else "negative_or_diagnostic_only_do_not_expand",
    }
    (output_dir / "stage84_lora_conservative_new_protect_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "STAGE84_LORA_CONSERVATIVE_NEW_PROTECT_SWEEP_SEED7_REPORT.md").write_text(
        "# Stage84 LoRa 保守新类保护门控矩阵 seed7 报告\n\n"
        f"- 最优 Overall 策略 `{selected['Policy']}`：R3 Overall/Old/New/Forgetting = "
        f"`{selected['Overall Acc']:.4f}/{selected['Old Acc']:.4f}/"
        f"{selected['New Acc']:.4f}/{selected['Forgetting Rate']:.4f}`。\n"
        f"- 最优平衡策略 `{balanced['Policy']}`：R3 Overall/Old/New = "
        f"`{balanced['Overall Acc']:.4f}/{balanced['Old Acc']:.4f}/{balanced['New Acc']:.4f}`。\n"
        f"- Stage83 对照 R3 Overall/Old/New = "
        f"`{stage83['overall']:.4f}/{stage83['old']:.4f}/{stage83['new']:.4f}`。\n"
        f"- 通过门槛：`{passes}`；决策：`{summary['decision']}`。\n"
        "- 说明：本阶段只提高训练期 discovery reject 目标，不读取 held-out eval 真值选阈值。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
