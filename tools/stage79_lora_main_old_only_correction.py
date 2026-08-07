"""Stage79：LoRa 主模型旧类预测的保守修正验证。

Stage77/78 的共同问题是：门控一旦判断样本“像旧类来源”，就会把预测切到
旧类专家，因此大量真实新类样本被吸回旧类，New Acc 明显塌缩。本阶段把路由
规则改得更保守：只有主 CIL 模型自己已经预测为旧类时，才允许旧类专家在旧
类内部重判；主模型已经预测为当前/历史新类的样本完全不改。

严格边界：
* 旧类专家和门控只使用当前日期旧设备 IQ_1-7 标注校准样本，以及当前轮
  discovery/enrollment 样本；
* 路由时只读取主模型 logits/embedding 与训练期门控，不读取 held-out 标签；
* IQ_8-10 held-out 标签只在最后计算指标时读取。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
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
from tools.stage77_lora_old_new_expert_gate import _checkpoint_for_round  # noqa: E402
from tools.stage78_lora_source_aware_gate import (  # noqa: E402
    _extract_outputs,
    _fit_source_gate,
    _old_prototypes,
    _source_features,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出逐轮指标，保持和前序 Stage 的轻量文本结果格式一致。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["Round"])
        writer.writeheader()
        writer.writerows(rows)


def _predict_with_main_old_only_correction(
    eval_logits: np.ndarray,
    eval_features: np.ndarray,
    old_logits: np.ndarray,
    old_labels: np.ndarray,
    prototypes: np.ndarray,
    gate: LogisticRegression,
    threshold: float,
    old_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """只修正主模型已判旧类的样本，避免把主模型已判新类的样本吸回旧类。

    返回值包含：
    * 修正后的预测；
    * 实际进入旧专家的样本 mask；
    * 主模型原始预测为旧类的样本 mask，用来衡量本轮修正上限。
    """

    logits = np.asarray(eval_logits, dtype=np.float32)
    main_pred = logits.argmax(axis=1).astype(np.int64)
    main_old_mask = main_pred < int(old_end)

    # 旧类专家只学习当前日期旧设备校准样本的 logits->旧类标签映射。
    # 它只能在旧类内部给出类别，不负责判断未知/新类来源。
    old_expert = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0)
    old_expert.fit(np.asarray(old_logits, dtype=np.float32), np.asarray(old_labels, dtype=np.int64))
    old_pred = old_expert.predict(logits).astype(np.int64)

    old_index = int(np.where(gate.classes_ == 1)[0][0])
    gate_x = _source_features(eval_logits, eval_features, prototypes)
    old_probability = gate.predict_proba(gate_x)[:, old_index]

    # Stage79 的关键约束：门控只能在“主模型已经判旧”的子集里生效。
    # 这样真实新类样本若主模型已判为新类，就不会被后处理改坏。
    route_old = main_old_mask & (old_probability >= float(threshold))
    corrected = main_pred.copy()
    corrected[route_old] = old_pred[route_old]
    return corrected, route_old, main_old_mask


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage79 LoRa 主模型旧类预测保守修正")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--stage76-save-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--transmissions", default="1,2,3,4,5,6,7")
    parser.add_argument("--symbols-per-transmission", type=int, default=28)
    parser.add_argument("--decimation", type=int, default=1)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage76_dir = Path(args.stage76_save_dir).resolve()
    transmissions = [int(value) for value in args.transmissions.split(",") if value.strip()]
    protocol = load_lora25_diffdays_3round(Path(args.dataset_path))
    rows: list[dict[str, Any]] = []
    initial_reference = None
    main_initial_reference = None

    for round_index in range(1, 4):
        stage = f"after_r{round_index}"
        bounds = _stage_bounds(stage)
        seen_classes = int(bounds["seen_classes"])
        old_end = int(bounds["old_end"])
        model = _load_lora_stage_model(
            _checkpoint_for_round(stage76_dir, round_index),
            seen_classes=seen_classes,
            feat_dim=args.feat_dim,
            device=args.device,
        )

        # 当前日期旧设备 IQ_1-7 是训练期可用校准数据；不读取 IQ_8-10。
        calibration_x, calibration_y, _ = _build_calibration_x(
            Path(args.raw_dir),
            day=round_index + 1,
            old_end=old_end,
            transmissions=transmissions,
            symbols_per_transmission=args.symbols_per_transmission,
            decimation=args.decimation,
            representation="raw",
        )
        calibration_features, calibration_logits = _extract_outputs(
            model, calibration_x, args.batch_size, args.device
        )

        discovery_key = f"day{round_index + 1}_unknown_round{round_index}"
        discovery_features, discovery_logits = _extract_outputs(
            model,
            np.asarray(protocol[discovery_key]["X"]),
            args.batch_size,
            args.device,
        )
        prototypes = _old_prototypes(calibration_features, calibration_y, old_end)
        gate, threshold, gate_quality = _fit_source_gate(
            calibration_logits,
            calibration_features,
            discovery_logits,
            discovery_features,
            prototypes,
            args.seed + round_index,
        )

        eval_key = f"day{round_index + 1}_eval_after_r{round_index}"
        eval_stage = protocol[eval_key]
        eval_features, eval_logits = _extract_outputs(
            model, np.asarray(eval_stage["X"]), args.batch_size, args.device
        )
        y_true = np.asarray(eval_stage["y"])
        main_prediction = np.asarray(eval_logits, dtype=np.float32).argmax(axis=1).astype(np.int64)
        prediction, route_old, main_old_mask = _predict_with_main_old_only_correction(
            eval_logits,
            eval_features,
            calibration_logits,
            calibration_y,
            prototypes,
            gate,
            threshold,
            old_end,
        )

        main_metrics = _evaluate(stage, y_true, main_prediction, main_initial_reference)
        if main_initial_reference is None:
            main_initial_reference = float(main_metrics["initial_known"])
        metrics = _evaluate(stage, y_true, prediction, initial_reference)
        if initial_reference is None:
            initial_reference = float(metrics["initial_known"])

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
                "Old Route Fraction": float(route_old.mean()),
                "Main Old Prediction Fraction": float(main_old_mask.mean()),
                "Gate Threshold": float(threshold),
                "Gate Old Source Recall": gate_quality["old_source_recall"],
                "Gate Discovery Reject": gate_quality["discovery_reject"],
            }
        )

    _write_csv(output_dir / "stage79_main_old_only_correction_results.csv", rows)
    r3 = rows[-1]
    reference = {"overall": 0.2810, "old": 0.2220, "new": 0.5167, "forgetting": 0.1726}
    summary = {
        "method": "main_old_prediction_only_correction",
        "protocol": (
            "old calibration IQ_1-7 + source gate, but correction only when "
            "main CIL prediction is already an old class; IQ_8-10 labels report-only"
        ),
        "stage76_save_dir": str(stage76_dir),
        "r3": r3,
        "delta_vs_stage48_seed7": {
            "overall": r3["Overall Acc"] - reference["overall"],
            "old": r3["Old Acc"] - reference["old"],
            "new": r3["New Acc"] - reference["new"],
            "forgetting": r3["Forgetting Rate"] - reference["forgetting"],
        },
        "delta_vs_main_stage76": {
            "overall": r3["Overall Acc"] - r3["Main Overall Acc"],
            "old": r3["Old Acc"] - r3["Main Old Acc"],
            "new": r3["New Acc"] - r3["Main New Acc"],
            "forgetting": r3["Forgetting Rate"] - r3["Main Forgetting Rate"],
        },
        "gate": (
            r3["Overall Acc"] >= reference["overall"] + 0.01
            and r3["Old Acc"] >= reference["old"]
            and r3["New Acc"] >= reference["new"] - 0.02
        ),
    }
    (output_dir / "stage79_main_old_only_correction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "STAGE79_LORA_MAIN_OLD_ONLY_CORRECTION_SEED7_REPORT.md").write_text(
        "# Stage79 LoRa 主模型旧类预测保守修正 seed7 报告\n\n"
        f"- R3 Overall/Old/New/Forgetting: `{r3['Overall Acc']:.4f}/"
        f"{r3['Old Acc']:.4f}/{r3['New Acc']:.4f}/{r3['Forgetting Rate']:.4f}`\n"
        f"- 主模型原始 R3 Overall/Old/New: `{r3['Main Overall Acc']:.4f}/"
        f"{r3['Main Old Acc']:.4f}/{r3['Main New Acc']:.4f}`\n"
        f"- 旧修正比例：`{r3['Old Route Fraction']:.4f}`；"
        f"主模型判旧比例：`{r3['Main Old Prediction Fraction']:.4f}`；"
        f"门控阈值：`{r3['Gate Threshold']:.2f}`。\n"
        f"- 通过门槛：`{summary['gate']}`。\n"
        "- 关键限制：主模型已判新类的样本完全不修改，用来保护 New Acc。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
