"""阶段 1 IGCD strict 最小入口 smoke test。

该脚本用于验证 IGCD-style baseline 是否能按本项目 60/10/30 strict
协议接入发现适配器契约。它使用合成多时间步数据模拟 WiSig R1/R2/R3：

- 已知类标签只存在于 Day1 train/validation；
- 每个增量轮次只把当前 discovery 特征作为未标注输入；
- 隐藏新类标签只在 predict 之后用于事后 NMI/ARI/purity 指标。

这不是完整 IGCD 论文复现，而是正式适配前的最小运行入口和无泄漏
审计证据。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 允许直接从项目外调用脚本，同时保持项目内部 import 路径稳定。
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.clustering import cluster_purity
from utils.discovery_adapter_contract import DiscoveryAdapterInput, validate_discovery_output
from utils.igcd_minimal_adapter import IGCDMinimalDiscoveryAdapter


def _make_centers(rng: np.random.Generator, count: int, dim: int, offset: float) -> np.ndarray:
    """生成彼此可分的合成类中心，用于快速 smoke test。"""
    base = rng.normal(0.0, 1.0, size=(count, dim))
    trend = np.linspace(0.0, offset, count, dtype=np.float32)[:, None]
    return base + trend + offset


def build_synthetic_time_steps(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[str, np.ndarray, np.ndarray]]]:
    """构造 Day1 已知 train/validation 和三轮未标注新类。

    返回的每轮 hidden labels 只供报告指标使用，不会写入
    `DiscoveryAdapterInput`。
    """
    rng = np.random.default_rng(seed)
    feature_dim = 24
    known_classes = 5
    novel_per_round = 3
    known_train_per_class = 36
    known_val_per_class = 12
    novel_per_class = 32

    known_centers = _make_centers(rng, known_classes, feature_dim, offset=0.0)
    known_train = []
    known_train_y = []
    known_val = []
    known_val_y = []
    for class_id, center in enumerate(known_centers):
        known_train.append(center + rng.normal(0.0, 0.25, size=(known_train_per_class, feature_dim)))
        known_train_y.extend([class_id] * known_train_per_class)
        known_val.append(center + rng.normal(0.0, 0.25, size=(known_val_per_class, feature_dim)))
        known_val_y.extend([class_id] * known_val_per_class)

    rounds: list[tuple[str, np.ndarray, np.ndarray]] = []
    for round_index in range(3):
        centers = _make_centers(rng, novel_per_round, feature_dim, offset=4.0 + round_index * 1.5)
        features = []
        hidden_labels = []
        for class_id, center in enumerate(centers):
            features.append(center + rng.normal(0.0, 0.25, size=(novel_per_class, feature_dim)))
            hidden_labels.extend([class_id] * novel_per_class)
        rounds.append(
            (
                f"R{round_index + 1}",
                np.vstack(features).astype(np.float32),
                np.asarray(hidden_labels, dtype=np.int64),
            )
        )

    return (
        np.vstack(known_train).astype(np.float32),
        np.asarray(known_train_y, dtype=np.int64),
        np.vstack(known_val).astype(np.float32),
        np.asarray(known_val_y, dtype=np.int64),
        rounds,
    )


def run_smoke(seed: int) -> dict[str, Any]:
    """运行三轮 IGCD strict smoke test 并返回 JSON 安全报告。"""
    known_x, known_y, val_x, val_y, rounds = build_synthetic_time_steps(seed)
    rows: list[dict[str, Any]] = []

    for round_name, unlabeled_x, hidden_y in rounds:
        data = DiscoveryAdapterInput(
            known_features=known_x,
            known_labels=known_y,
            validation_features=val_x,
            validation_labels=val_y,
            unlabeled_features=unlabeled_x,
            round_name=round_name,
            expected_new_classes=3,
        )
        adapter = IGCDMinimalDiscoveryAdapter(random_state=seed)
        adapter.fit(data)
        output = adapter.predict(data)
        validate_discovery_output(output, expected_samples=unlabeled_x.shape[0])
        rows.append(
            {
                "round": round_name,
                "adapter": adapter.name,
                "estimated_new_classes": int(output.estimated_new_classes),
                "samples": int(output.cluster_labels.shape[0]),
                "confidence_mean": float(np.mean(output.confidence)),
                "confidence_min": float(np.min(output.confidence)),
                "nmi_posthoc": float(normalized_mutual_info_score(hidden_y, output.cluster_labels)),
                "ari_posthoc": float(adjusted_rand_score(hidden_y, output.cluster_labels)),
                "purity_posthoc": float(cluster_purity(hidden_y, output.cluster_labels, ignore_noise=False)),
                "uses_unknown_true_labels": bool(output.diagnostics.get("uses_unknown_true_labels")),
                "uses_eval_set": bool(output.diagnostics.get("uses_eval_set")),
                "diagnostics": output.diagnostics,
            }
        )

    return {
        "schema_version": "stage1_igcd_strict_entry_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "protocol_mapping": {
            "source": "IGCD time steps",
            "target": "WiSig strict R1/R2/R3 with Day1 60/10/30 boundary",
            "allowed_inputs": "known train/validation embeddings + current round unlabeled discovery embeddings",
            "forbidden_inputs": "unknown true labels for fit/predict and held-out eval set",
        },
        "implementation_level": "minimal_strict_entry_not_full_paper_reproduction",
        "rows": rows,
        "next_integration_step": (
            "将该入口接入真实 WiSig frozen embeddings；若指标和协议边界成立，"
            "再评估是否复现完整 IGCD 的采样与非参数分类组件。"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行阶段 1 IGCD strict 最小入口 smoke test。")
    parser.add_argument("--seed", type=int, default=7, help="合成数据和 KMeans 随机种子。")
    parser.add_argument("--output", default="results/stage1/stage1_igcd_strict_entry.json", help="输出 JSON 路径。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_smoke(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"阶段 1 IGCD strict 最小入口：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
