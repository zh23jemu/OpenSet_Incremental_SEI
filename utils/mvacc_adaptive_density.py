"""MV-ACC 轮次自适应密度选择的严格无标签门控。

该模块只接收 discovery 特征上计算出的结构指标和两个候选聚类标签。
真实设备标签、协议中的真实新类数量以及 held-out evaluation 指标均不属于
接口，因此调用方无法在密度选择阶段意外引入测试标签泄漏。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score


@dataclass(frozen=True)
class AdaptiveDensityThresholds:
    """保守切换门槛；任何一项不满足时都继续使用锚点配置。"""

    max_added_clusters: int = 2
    min_silhouette_gain: float = 0.04
    max_confidence_drop: float = 0.03
    max_cv_increase: float = 0.10
    max_small_cluster_fraction_increase: float = 0.10
    min_partition_agreement: float = 0.50


def _finite_delta(candidate: float, anchor: float) -> float:
    """计算有限差值；缺失指标返回 NaN，并由门控显式拒绝候选。"""

    if not np.isfinite(candidate) or not np.isfinite(anchor):
        return float("nan")
    return float(candidate - anchor)


def select_adaptive_density_candidate(
    anchor: dict[str, Any],
    candidate: dict[str, Any],
    thresholds: AdaptiveDensityThresholds,
) -> dict[str, Any]:
    """在两个密度候选之间做保守、可审计的无标签选择。

    参数字典必须包含 ``ratio``、``labels``、``cluster_count``、
    ``silhouette``、``confidence``、``cluster_size_cv`` 和
    ``small_cluster_fraction``。候选只有在改善分离度、未明显降低置信度或
    簇均衡性、与锚点分区仍保持基本一致且最多新增少量簇时才会被采用。
    """

    anchor_labels = np.asarray(anchor["labels"], dtype=np.int64)
    candidate_labels = np.asarray(candidate["labels"], dtype=np.int64)
    if anchor_labels.shape != candidate_labels.shape:
        raise ValueError("锚点与候选聚类标签必须对应同一批 discovery 样本。")

    cluster_delta = int(candidate["cluster_count"]) - int(anchor["cluster_count"])
    silhouette_delta = _finite_delta(float(candidate["silhouette"]), float(anchor["silhouette"]))
    confidence_delta = _finite_delta(float(candidate["confidence"]), float(anchor["confidence"]))
    cv_delta = _finite_delta(float(candidate["cluster_size_cv"]), float(anchor["cluster_size_cv"]))
    small_fraction_delta = _finite_delta(
        float(candidate["small_cluster_fraction"]),
        float(anchor["small_cluster_fraction"]),
    )
    agreement = float(adjusted_rand_score(anchor_labels, candidate_labels))

    checks = {
        "adds_limited_clusters": 0 < cluster_delta <= int(thresholds.max_added_clusters),
        "silhouette_improves": (
            np.isfinite(silhouette_delta)
            and silhouette_delta >= float(thresholds.min_silhouette_gain)
        ),
        "confidence_preserved": (
            np.isfinite(confidence_delta)
            and confidence_delta >= -float(thresholds.max_confidence_drop)
        ),
        "cluster_balance_preserved": (
            np.isfinite(cv_delta)
            and cv_delta <= float(thresholds.max_cv_increase)
        ),
        "small_cluster_fraction_preserved": (
            np.isfinite(small_fraction_delta)
            and small_fraction_delta <= float(thresholds.max_small_cluster_fraction_increase)
        ),
        "partition_agreement_preserved": agreement >= float(thresholds.min_partition_agreement),
    }
    use_candidate = bool(all(checks.values()))
    rejected_by = [name for name, passed in checks.items() if not passed]

    return {
        "selected_ratio": float(candidate["ratio"] if use_candidate else anchor["ratio"]),
        "selected_source": "candidate" if use_candidate else "anchor",
        "candidate_accepted": use_candidate,
        "rejected_by": rejected_by,
        "cluster_count_delta": cluster_delta,
        "silhouette_delta": silhouette_delta,
        "confidence_delta": confidence_delta,
        "cluster_size_cv_delta": cv_delta,
        "small_cluster_fraction_delta": small_fraction_delta,
        "partition_agreement_ari": agreement,
        "checks": checks,
    }
