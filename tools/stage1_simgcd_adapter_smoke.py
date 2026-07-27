"""阶段 1 SimGCD 式发现适配器 smoke test。

该脚本使用合成特征验证最小适配器能完成 fit/predict、输出契约校验和
基础聚类指标计算。合成数据的隐藏标签只用于 smoke test 的事后指标，
不传入 `DiscoveryAdapterInput`，用于模拟项目正式实验中的无泄漏边界。
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
    # 允许从任意工作目录直接运行 tools 脚本，同时不要求用户手动设置 PYTHONPATH。
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.clustering import cluster_purity
from utils.discovery_adapter_contract import DiscoveryAdapterInput, validate_discovery_output
from utils.simgcd_discovery_adapter import SimGCDStyleDiscoveryAdapter


def build_synthetic_case(seed: int) -> tuple[DiscoveryAdapterInput, np.ndarray]:
    """构造一个小型合成发现任务。

    已知类与未标注新类位于不同中心附近，便于快速检查适配器是否能
    输出合理簇标签。`hidden_unlabeled_labels` 仅返回给报告指标，不进入
    适配器输入对象。
    """
    rng = np.random.default_rng(seed)
    feature_dim = 16
    known_classes = 4
    novel_classes = 3
    known_per_class = 32
    novel_per_class = 40

    known_centers = rng.normal(0.0, 1.0, size=(known_classes, feature_dim))
    novel_centers = rng.normal(0.0, 1.0, size=(novel_classes, feature_dim)) + 4.0

    known_features = []
    known_labels = []
    for class_id, center in enumerate(known_centers):
        known_features.append(center + rng.normal(0.0, 0.25, size=(known_per_class, feature_dim)))
        known_labels.extend([class_id] * known_per_class)

    unlabeled_features = []
    hidden_unlabeled_labels = []
    for class_id, center in enumerate(novel_centers):
        unlabeled_features.append(center + rng.normal(0.0, 0.25, size=(novel_per_class, feature_dim)))
        hidden_unlabeled_labels.extend([class_id] * novel_per_class)

    data = DiscoveryAdapterInput(
        known_features=np.vstack(known_features).astype(np.float32),
        known_labels=np.asarray(known_labels, dtype=np.int64),
        unlabeled_features=np.vstack(unlabeled_features).astype(np.float32),
        round_name="synthetic_r1",
        expected_new_classes=novel_classes,
    )
    return data, np.asarray(hidden_unlabeled_labels, dtype=np.int64)


def run_smoke(seed: int) -> dict[str, Any]:
    """运行 smoke test 并返回 JSON 安全报告。"""
    data, hidden_labels = build_synthetic_case(seed)
    adapter = SimGCDStyleDiscoveryAdapter(random_state=seed)
    adapter.fit(data)
    output = adapter.predict(data)
    validate_discovery_output(output, expected_samples=data.unlabeled_features.shape[0])

    return {
        "schema_version": "stage1_simgcd_adapter_smoke_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "adapter": adapter.name,
        "seed": int(seed),
        "estimated_new_classes": int(output.estimated_new_classes),
        "samples": int(output.cluster_labels.shape[0]),
        "confidence_mean": float(np.mean(output.confidence)),
        "confidence_min": float(np.min(output.confidence)),
        "nmi_posthoc": float(normalized_mutual_info_score(hidden_labels, output.cluster_labels)),
        "ari_posthoc": float(adjusted_rand_score(hidden_labels, output.cluster_labels)),
        "purity_posthoc": float(cluster_purity(hidden_labels, output.cluster_labels, ignore_noise=False)),
        "diagnostics": output.diagnostics,
        "leakage_note": "hidden labels are used only after predict for synthetic smoke metrics",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行阶段 1 SimGCD 式发现适配器 smoke test。")
    parser.add_argument("--seed", type=int, default=7, help="合成数据和 KMeans 随机种子。")
    parser.add_argument("--output", default="results/stage1/stage1_simgcd_adapter_smoke.json", help="输出 JSON 路径。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_smoke(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"阶段 1 SimGCD 式适配器 smoke test：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
