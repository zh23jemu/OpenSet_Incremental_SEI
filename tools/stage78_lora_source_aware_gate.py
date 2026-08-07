"""Stage78：LoRa 表征来源感知的旧/新专家门控验证。

Stage77 的问题是：仅依赖分类 logits 的门控把大量真实新类样本路由到旧类
专家，导致 Old 提升、New 严重下降。本阶段加入当前增量模型 embedding 到
旧类原型的相似度，构造“分类输出 + 表征来源”联合门控。

严格边界：
* 旧类原型只由 Day2-4 的 IQ_1-7 标注校准样本构造；
* 门控训练只使用旧类校准样本和当前轮 discovery 样本；
* IQ_8-10 的标签只在最终计算指标时读取，不参与原型、门控或阈值选择。
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

from tools.stage69_lora_old_day_calibration_report import (  # noqa: E402
    _build_calibration_x,
    _evaluate,
    _load_lora_stage_model,
    _stage_bounds,
)
from tools.stage77_lora_old_new_expert_gate import (  # noqa: E402
    _checkpoint_for_round,
    _fit_gate,
    _split_indices,
)
from datasets.lora25_strict_loader import load_lora25_diffdays_3round  # noqa: E402


def _extract_outputs(model, x: np.ndarray, batch_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """批量提取 embedding 和 logits，避免一次性占满显存。"""

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(
        TensorDataset(torch.as_tensor(np.asarray(x, dtype=np.float32))),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            feat, out = model(xb.to(device))
            features.append(feat.detach().cpu().numpy())
            logits.append(out.detach().cpu().numpy())
    return np.concatenate(features, axis=0), np.concatenate(logits, axis=0)


def _old_prototypes(features: np.ndarray, labels: np.ndarray, old_end: int) -> np.ndarray:
    """按旧类标签构造归一化类中心；缺失类用全体均值兜底。"""

    z = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    global_mean = z.mean(axis=0)
    prototypes = []
    for class_id in range(int(old_end)):
        mask = y == class_id
        center = z[mask].mean(axis=0) if np.any(mask) else global_mean
        center = center / max(float(np.linalg.norm(center)), 1e-8)
        prototypes.append(center)
    return np.asarray(prototypes, dtype=np.float32)


def _source_features(logits: np.ndarray, features: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """拼接 logits 统计与旧类原型相似度统计，形成来源判别特征。"""

    values = np.asarray(logits, dtype=np.float32)
    z = np.asarray(features, dtype=np.float32)
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
    similarity = z @ prototypes.T
    topk = np.sort(similarity, axis=1)[:, -min(3, similarity.shape[1]):]
    shifted = values - values.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-8)
    entropy = -(probs * np.log(np.maximum(probs, 1e-8))).sum(axis=1, keepdims=True)
    top2 = np.partition(values, kth=max(0, values.shape[1] - 2), axis=1)[:, -2:]
    logit_margin = (top2[:, 1] - top2[:, 0]).reshape(-1, 1)
    proto_margin = (topk[:, -1] - topk[:, -2]).reshape(-1, 1) if topk.shape[1] >= 2 else topk[:, -1:]
    return np.concatenate(
        [values, entropy, logit_margin, topk, proto_margin],
        axis=1,
    ).astype(np.float32)


def _fit_source_gate(
    old_logits: np.ndarray,
    old_features: np.ndarray,
    discovery_logits: np.ndarray,
    discovery_features: np.ndarray,
    prototypes: np.ndarray,
    seed: int,
) -> tuple[LogisticRegression, float, dict[str, float]]:
    """用旧类/当前 discovery 来源训练门控，并在训练期留出集选阈值。"""

    old_x = _source_features(old_logits, old_features, prototypes)
    discovery_x = _source_features(discovery_logits, discovery_features, prototypes)
    old_train, old_valid = _split_indices(len(old_x), seed)
    new_train, new_valid = _split_indices(len(discovery_x), seed + 1)
    train_x = np.concatenate([old_x[old_train], discovery_x[new_train]], axis=0)
    train_y = np.concatenate(
        [np.ones(len(old_train), dtype=np.int64), np.zeros(len(new_train), dtype=np.int64)]
    )
    gate = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    gate.fit(train_x, train_y)

    valid_x = np.concatenate([old_x[old_valid], discovery_x[new_valid]], axis=0)
    valid_y = np.concatenate(
        [np.ones(len(old_valid), dtype=np.int64), np.zeros(len(new_valid), dtype=np.int64)]
    )
    old_index = int(np.where(gate.classes_ == 1)[0][0])
    valid_prob = gate.predict_proba(valid_x)[:, old_index]
    scored = []
    for threshold in np.linspace(0.10, 0.90, 17):
        pred = (valid_prob >= threshold).astype(np.int64)
        old_recall = float(np.mean(pred[valid_y == 1] == 1))
        discovery_reject = float(np.mean(pred[valid_y == 0] == 0))
        scored.append((min(old_recall, discovery_reject), old_recall + discovery_reject, threshold, old_recall, discovery_reject))
    _, _, threshold, old_recall, discovery_reject = max(scored)
    full_x = np.concatenate([old_x, discovery_x], axis=0)
    full_y = np.concatenate(
        [np.ones(len(old_x), dtype=np.int64), np.zeros(len(discovery_x), dtype=np.int64)]
    )
    gate.fit(full_x, full_y)
    return gate, float(threshold), {
        "old_source_recall": old_recall,
        "discovery_reject": discovery_reject,
    }


def _predict_with_source_gate(
    eval_logits: np.ndarray,
    eval_features: np.ndarray,
    old_logits: np.ndarray,
    old_labels: np.ndarray,
    old_features: np.ndarray,
    prototypes: np.ndarray,
    gate: LogisticRegression,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """训练旧类专家，并用来源感知门控决定是否切换到旧专家。"""

    old_expert = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0)
    old_expert.fit(np.asarray(old_logits, dtype=np.float32), np.asarray(old_labels, dtype=np.int64))
    main_pred = np.asarray(eval_logits, dtype=np.float32).argmax(axis=1).astype(np.int64)
    old_pred = old_expert.predict(np.asarray(eval_logits, dtype=np.float32)).astype(np.int64)
    old_index = int(np.where(gate.classes_ == 1)[0][0])
    gate_x = _source_features(eval_logits, eval_features, prototypes)
    old_probability = gate.predict_proba(gate_x)[:, old_index]
    route_old = old_probability >= float(threshold)
    main_pred[route_old] = old_pred[route_old]
    return main_pred, route_old


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["Round"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage78 LoRa 表征来源感知双专家门控")
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
        seen_classes = int(bounds["seen_classes"])
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
        prediction, route_old = _predict_with_source_gate(
            eval_logits,
            eval_features,
            calibration_logits,
            calibration_y,
            calibration_features,
            prototypes,
            gate,
            threshold,
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

    _write_csv(output_dir / "stage78_source_aware_gate_results.csv", rows)
    r3 = rows[-1]
    reference = {"overall": 0.2810, "old": 0.2220, "new": 0.5167, "forgetting": 0.1726}
    summary = {
        "method": "source_aware_old_expert_plus_pseudonew_head",
        "protocol": "old calibration IQ_1-7 + embedding-to-old-prototype source gate; IQ_8-10 labels report-only",
        "r3": r3,
        "delta_vs_stage48_seed7": {
            "overall": r3["Overall Acc"] - reference["overall"],
            "old": r3["Old Acc"] - reference["old"],
            "new": r3["New Acc"] - reference["new"],
            "forgetting": r3["Forgetting Rate"] - reference["forgetting"],
        },
        "gate": (
            r3["Overall Acc"] >= reference["overall"] + 0.01
            and r3["Old Acc"] >= reference["old"]
            and r3["New Acc"] >= reference["new"] - 0.02
        ),
    }
    (output_dir / "stage78_source_aware_gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "STAGE78_LORA_SOURCE_AWARE_GATE_SEED7_REPORT.md").write_text(
        "# Stage78 LoRa 表征来源感知门控 seed7 报告\n\n"
        f"- R3 Overall/Old/New/Forgetting: `{r3['Overall Acc']:.4f}/"
        f"{r3['Old Acc']:.4f}/{r3['New Acc']:.4f}/{r3['Forgetting Rate']:.4f}`\n"
        f"- 旧路由比例：`{r3['Old Route Fraction']:.4f}`；门控阈值：`{r3['Gate Threshold']:.2f}`。\n"
        f"- 通过门槛：`{summary['gate']}`。\n"
        "- 门控加入 embedding 到旧类原型的相似度统计；IQ_8-10 仅用于最终指标。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
