"""无标签 discovery 特征适配工具。

该模块用于 ADS-B / LoRa 低分风险的前端重设计：在进入 HDBSCAN/MV-ACC
或 GPCC 聚类之前，只使用 Day1 已知类训练特征和当前轮 discovery 特征，
对当前轮 deep embedding 做轻量变换。它不读取未知类真实标签，也不读取
held-out evaluation 样本，因此可以用于 strict discovery-only 审计。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize


@dataclass(frozen=True)
class DiscoveryFeatureAdaptationResult:
    """特征适配结果。

    attributes
    ----------
    features:
        适配后的 discovery 特征，行数与输入 discovery 样本一致。
    diagnostics:
        可直接写入 clustering CSV 的无标签诊断信息。
    """

    features: np.ndarray
    diagnostics: dict[str, Any]


def _as_float_matrix(x: np.ndarray, name: str) -> np.ndarray:
    """将输入转换为二维 float32 矩阵，并清理 NaN/Inf。"""

    z = np.asarray(x, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"{name} must be a 2D feature matrix, got shape={z.shape}.")
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _l2(x: np.ndarray) -> np.ndarray:
    """样本级 L2 归一化，避免某个维度或某个 view 的尺度支配聚类。"""

    return normalize(np.asarray(x, dtype=np.float32), norm="l2", axis=1).astype(np.float32)


def _known_prototypes(known_z: np.ndarray, known_y: np.ndarray) -> np.ndarray:
    """按已知类标签计算类中心；这里只使用 Day1 train/replay 的真实已知标签。"""

    labels = np.asarray(known_y, dtype=np.int64)
    protos = []
    for class_id in sorted(np.unique(labels).tolist()):
        mask = labels == int(class_id)
        if np.any(mask):
            protos.append(known_z[mask].mean(axis=0))
    if not protos:
        return np.zeros((0, known_z.shape[1]), dtype=np.float32)
    return _l2(np.stack(protos, axis=0))


def _mutual_neighbor_smooth(discovery_z: np.ndarray, k: int, weight: float) -> np.ndarray:
    """只在当前轮 discovery 内做局部结构平滑。

    该操作使用 cosine 近邻构造无标签局部平均。它不会改变样本数，也不会
    产生伪标签；作用是降低单个 embedding 噪声对后续图聚类的影响。
    """

    z = _l2(discovery_z)
    n = int(z.shape[0])
    k_eff = max(1, min(int(k), n - 1))
    weight = float(np.clip(weight, 0.0, 1.0))
    if n <= 2 or k_eff <= 0 or weight <= 0:
        return z.astype(np.float32)

    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine")
    nn.fit(z)
    indices = nn.kneighbors(z, return_distance=False)[:, 1:]
    smoothed = np.empty_like(z)
    for row_idx, nbr_idx in enumerate(indices):
        local_mean = z[nbr_idx].mean(axis=0)
        smoothed[row_idx] = (1.0 - weight) * z[row_idx] + weight * local_mean
    return _l2(smoothed)


def adapt_discovery_features(
    known_features: np.ndarray | None,
    known_labels: np.ndarray | None,
    discovery_features: np.ndarray,
    *,
    method: str = "none",
    smooth_k: int = 12,
    smooth_weight: float = 0.20,
    repulsion_weight: float = 0.15,
) -> DiscoveryFeatureAdaptationResult:
    """对当前轮 discovery 特征做严格无标签适配。

    参数
    ----
    known_features / known_labels:
        Day1 已知类训练特征与标签。仅用于拟合标准化统计和已知类原型；
        不允许传入未知类标签或 held-out evaluation 特征。
    discovery_features:
        当前轮待聚类样本的 deep embedding。
    method:
        ``none`` 保持原始行为；``mn_smooth`` 只做 discovery 局部平滑；
        ``proto_repulse`` 先从最近已知类原型方向轻微排斥，再做局部平滑。
    smooth_k / smooth_weight:
        当前轮近邻平滑的邻居数和混合权重。
    repulsion_weight:
        ``proto_repulse`` 的已知类原型排斥强度。
    """

    method = str(method or "none").lower()
    discovery = _as_float_matrix(discovery_features, "discovery_features")
    diagnostics: dict[str, Any] = {
        "Discovery Feature Adapter": method,
        "DFA Smooth K": int(smooth_k),
        "DFA Smooth Weight": float(smooth_weight),
        "DFA Repulsion Weight": float(repulsion_weight),
        "DFA Known Samples": 0,
        "DFA Known Classes": 0,
    }

    if method == "none":
        return DiscoveryFeatureAdaptationResult(_l2(discovery), diagnostics)
    if method not in {"mn_smooth", "proto_repulse"}:
        raise ValueError(f"Unsupported discovery feature adapter: {method}")

    if known_features is None or known_labels is None or len(known_features) == 0:
        base = _l2(discovery)
    else:
        known = _as_float_matrix(known_features, "known_features")
        labels = np.asarray(known_labels, dtype=np.int64)
        if known.shape[1] != discovery.shape[1]:
            raise ValueError(
                "known_features and discovery_features must have the same dimension: "
                f"{known.shape[1]} vs {discovery.shape[1]}"
            )
        scaler = StandardScaler().fit(np.vstack([known, discovery]))
        known_z = _l2(scaler.transform(known))
        base = _l2(scaler.transform(discovery))
        diagnostics["DFA Known Samples"] = int(len(known_z))
        diagnostics["DFA Known Classes"] = int(np.unique(labels).size)

        if method == "proto_repulse":
            protos = _known_prototypes(known_z, labels)
            if len(protos) > 0 and float(repulsion_weight) > 0:
                sim = base @ protos.T
                nearest = np.argmax(sim, axis=1)
                strength = np.maximum(sim[np.arange(len(base)), nearest], 0.0)[:, None]
                base = _l2(base - float(repulsion_weight) * strength * protos[nearest])
                diagnostics["DFA Mean Nearest Known Cosine"] = float(np.mean(np.max(sim, axis=1)))
            else:
                diagnostics["DFA Mean Nearest Known Cosine"] = float("nan")

    adapted = _mutual_neighbor_smooth(base, k=int(smooth_k), weight=float(smooth_weight))
    diagnostics["DFA Output Dim"] = int(adapted.shape[1])
    return DiscoveryFeatureAdaptationResult(adapted.astype(np.float32), diagnostics)
