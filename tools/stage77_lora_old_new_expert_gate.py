"""Stage77：LoRa 旧类跨天专家与伪新类增量头分离验证。

Stage76 证明把当天旧类标注直接加入统一分类头，会提升旧类但明显挤压新类。
本工具不再修改 CIL 训练，而是把两种职责拆开：

* 旧类专家：只由当天旧设备 IQ_1-7 标注样本训练，负责旧设备内部识别；
* 增量主头：保留 Stage76/Stage48 的完整类别预测，负责伪新类及未被门控的样本；
* 二元门控：只用“旧类校准样本”和当前轮 discovery 样本训练，决定是否交给旧类专家。

严格边界：IQ_8-10 的真值只在最后计算指标时读取，绝不用于专家、门控或阈值选择。
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
    _extract_logits,
    _load_lora_stage_model,
    _stage_bounds,
)


def _split_indices(size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """按固定种子分出门控阈值选择集，避免在同一批训练样本上选阈值。"""

    if size < 4:
        indices = np.arange(size, dtype=np.int64)
        return indices, indices
    rng = np.random.default_rng(seed)
    indices = rng.permutation(size)
    cut = max(1, min(size - 1, int(round(size * 0.7))))
    return indices[:cut], indices[cut:]


def _gate_features(logits: np.ndarray) -> np.ndarray:
    """构造只依赖模型输出的门控特征，不接触评估标签。"""

    values = np.asarray(logits, dtype=np.float32)
    shifted = values - values.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
    entropy = -(probs * np.log(np.maximum(probs, 1e-8))).sum(axis=1, keepdims=True)
    top2 = np.partition(values, kth=max(0, values.shape[1] - 2), axis=1)[:, -2:]
    margin = (top2[:, 1] - top2[:, 0]).reshape(-1, 1)
    return np.concatenate([values, entropy, margin], axis=1).astype(np.float32)


def _fit_gate(
    old_logits: np.ndarray,
    discovery_logits: np.ndarray,
    seed: int,
) -> tuple[LogisticRegression, float, dict[str, float]]:
    """训练旧/新来源门控，并只在留出的训练期样本上选择保守阈值。"""

    old_features = _gate_features(old_logits)
    discovery_features = _gate_features(discovery_logits)
    old_train, old_valid = _split_indices(len(old_features), seed)
    new_train, new_valid = _split_indices(len(discovery_features), seed + 1)
    train_x = np.concatenate([old_features[old_train], discovery_features[new_train]], axis=0)
    train_y = np.concatenate(
        [np.ones(len(old_train), dtype=np.int64), np.zeros(len(new_train), dtype=np.int64)],
        axis=0,
    )
    gate = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, n_jobs=1)
    gate.fit(train_x, train_y)

    valid_x = np.concatenate([old_features[old_valid], discovery_features[new_valid]], axis=0)
    valid_y = np.concatenate(
        [np.ones(len(old_valid), dtype=np.int64), np.zeros(len(new_valid), dtype=np.int64)],
        axis=0,
    )
    valid_prob = gate.predict_proba(valid_x)[:, int(np.where(gate.classes_ == 1)[0][0])]
    # 阈值仅优化“旧类校准 / 当轮 discovery”来源的平衡准确率，不读取 held-out eval。
    candidates = np.linspace(0.10, 0.90, 17)
    scored = []
    for threshold in candidates:
        pred = (valid_prob >= threshold).astype(np.int64)
        old_recall = float(np.mean(pred[valid_y == 1] == 1))
        new_reject = float(np.mean(pred[valid_y == 0] == 0))
        scored.append((min(old_recall, new_reject), old_recall + new_reject, float(threshold), old_recall, new_reject))
    _, _, threshold, old_recall, new_reject = max(scored)

    full_x = np.concatenate([old_features, discovery_features], axis=0)
    full_y = np.concatenate(
        [np.ones(len(old_features), dtype=np.int64), np.zeros(len(discovery_features), dtype=np.int64)],
        axis=0,
    )
    gate.fit(full_x, full_y)
    return gate, threshold, {"old_source_recall": old_recall, "discovery_reject": new_reject}


def _apply_two_expert_gate(
    eval_logits: np.ndarray,
    old_logits: np.ndarray,
    old_labels: np.ndarray,
    gate: LogisticRegression,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """根据门控概率选择旧类专家或完整增量头，返回预测与旧路由掩码。"""

    old_expert = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0, n_jobs=1)
    old_expert.fit(np.asarray(old_logits, dtype=np.float32), np.asarray(old_labels, dtype=np.int64))
    main_pred = np.asarray(eval_logits, dtype=np.float32).argmax(axis=1).astype(np.int64)
    old_pred = old_expert.predict(np.asarray(eval_logits, dtype=np.float32)).astype(np.int64)
    old_class_index = int(np.where(gate.classes_ == 1)[0][0])
    old_probability = gate.predict_proba(_gate_features(eval_logits))[:, old_class_index]
    route_old = old_probability >= float(threshold)
    main_pred[route_old] = old_pred[route_old]
    return main_pred, route_old


def _checkpoint_for_round(save_dir: Path, round_index: int) -> Path:
    return save_dir / f"mvacc_cil_after_r{round_index}.pth"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["Round"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage77 LoRa 旧/新双专家门控验证")
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

    for round_index in range(1, 4):
        stage = f"after_r{round_index}"
        bounds = _stage_bounds(stage)
        seen_classes = int(bounds["seen_end"])
        old_end = int(bounds["old_end"])
        model = _load_lora_stage_model(
            _checkpoint_for_round(stage76_dir, round_index),
            seen_classes=seen_classes,
            feat_dim=args.feat_dim,
            device=args.device,
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
        calibration_logits = _extract_logits(model, calibration_x, args.batch_size, args.device)
        discovery_key = f"day{round_index + 1}_unknown_round{round_index}"
        discovery_logits = _extract_logits(
            model,
            np.asarray(protocol[discovery_key]["X"]),
            args.batch_size,
            args.device,
        )
        gate, threshold, gate_quality = _fit_gate(calibration_logits, discovery_logits, args.seed + round_index)

        eval_key = f"day{round_index + 1}_eval_after_r{round_index}"
        eval_stage = protocol[eval_key]
        eval_logits = _extract_logits(model, np.asarray(eval_stage["X"]), args.batch_size, args.device)
        prediction, route_old = _apply_two_expert_gate(
            eval_logits, calibration_logits, calibration_y, gate, threshold
        )
        metrics = _evaluate(stage, np.asarray(eval_stage["y"]), prediction, initial_reference)
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
                "Old Route Fraction": float(route_old.mean()),
                "Gate Threshold": float(threshold),
                "Gate Old Source Recall": gate_quality["old_source_recall"],
                "Gate Discovery Reject": gate_quality["discovery_reject"],
            }
        )

    _write_csv(output_dir / "stage77_two_expert_gate_results.csv", rows)
    r3 = rows[-1]
    stage48_seed7 = {"overall": 0.2810, "old": 0.2220, "new": 0.5167, "forgetting": 0.1726}
    summary = {
        "method": "old_day_expert_plus_pseudonew_head",
        "protocol": "old calibration IQ_1-7 + current discovery gate; IQ_8-10 labels report-only",
        "stage76_save_dir": str(stage76_dir),
        "r3": r3,
        "delta_vs_stage48_seed7": {
            "overall": r3["Overall Acc"] - stage48_seed7["overall"],
            "old": r3["Old Acc"] - stage48_seed7["old"],
            "new": r3["New Acc"] - stage48_seed7["new"],
            "forgetting": r3["Forgetting Rate"] - stage48_seed7["forgetting"],
        },
        "gate": (
            r3["Overall Acc"] >= stage48_seed7["overall"] + 0.01
            and r3["Old Acc"] >= stage48_seed7["old"]
            and r3["New Acc"] >= stage48_seed7["new"] - 0.02
        ),
    }
    (output_dir / "stage77_two_expert_gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "STAGE77_LORA_TWO_EXPERT_GATE_SEED7_REPORT.md").write_text(
        "# Stage77 LoRa 旧/新双专家门控 seed7 报告\n\n"
        f"- R3 Overall/Old/New/Forgetting: `{r3['Overall Acc']:.4f}/"
        f"{r3['Old Acc']:.4f}/{r3['New Acc']:.4f}/{r3['Forgetting Rate']:.4f}`\n"
        f"- 旧路由比例：`{r3['Old Route Fraction']:.4f}`；门控阈值：`{r3['Gate Threshold']:.2f}`。\n"
        f"- 通过门槛：`{summary['gate']}`。\n"
        "- IQ_8-10 仅在最终指标计算时使用；门控和旧类专家不读取其标签。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
