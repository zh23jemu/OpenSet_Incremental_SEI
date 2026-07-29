"""Graph-Prototype Controlled Clustering（GPCC）发现前端。

该模块实现一个严格无真值泄漏的目标类数约束聚类器，用来替代
HDBSCAN 作为 ADS-B / LoRa 低分风险的发现前端候选。GPCC 只读取当前轮
discovery 样本的多视图特征，以及实验协议在运行前已经公开声明的
每轮新增类别数 ``target_clusters``；不会读取未知真实标签或 held-out
evaluation 样本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class GPCCResult:
    """GPCC 的完整输出，便于实验入口生成统一报告。"""

    labels: np.ndarray
    confidence: np.ndarray
    diagnostics: dict[str, Any]


def _safe_normalize(x: np.ndarray) -> np.ndarray:
    """按行归一化并清理异常值，避免某一路特征尺度支配聚类。"""
    z = np.asarray(x, dtype=np.float32)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return normalize(z, norm="l2", axis=1).astype(np.float32)


def _dense_relabel(labels: np.ndarray) -> np.ndarray:
    """将任意聚类标签压缩为连续 0..K-1，并保证没有 noise 标签。"""
    labels = np.asarray(labels, dtype=np.int64)
    unique = sorted(int(v) for v in np.unique(labels))
    mapping = {old: new for new, old in enumerate(unique)}
    return np.asarray([mapping[int(v)] for v in labels], dtype=np.int64)


def _cluster_size_cv(labels: np.ndarray) -> float:
    """计算簇大小变异系数，作为 label-free 均衡性诊断。"""
    _, counts = np.unique(labels, return_counts=True)
    counts = counts.astype(np.float64)
    return float(np.std(counts) / (np.mean(counts) + 1e-8))


def _prototype_margin_confidence(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """基于最近/次近簇中心距离差构造 [0, 1] 置信度。

    距离最近中心明显优于次近中心时，样本更像一个稳定簇成员；当两个中心
    距离接近时，说明它靠近决策边界，置信度降低。
    """
    z = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    centers = []
    for cid in sorted(np.unique(labels).tolist()):
        centers.append(z[labels == int(cid)].mean(axis=0))
    centers = np.asarray(centers, dtype=np.float32)
    diff = z[:, None, :] - centers[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    if distances.shape[1] == 1:
        return np.ones(z.shape[0], dtype=np.float32)
    ordered = np.sort(distances, axis=1)
    margin = (ordered[:, 1] - ordered[:, 0]) / (ordered[:, 1] + 1e-8)
    return np.clip(margin, 0.0, 1.0).astype(np.float32)


def _silhouette_confidence(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """返回样本级 silhouette 置信度和整体 silhouette 诊断。"""
    labels = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels)) <= 1 or len(labels) <= len(np.unique(labels)):
        return np.ones(len(labels), dtype=np.float32), np.nan
    raw = silhouette_samples(features, labels, metric="euclidean")
    conf = np.clip((raw + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
    return conf, float(np.mean(raw))


def _candidate_score(features: np.ndarray, labels: np.ndarray, reference_labels: np.ndarray) -> dict[str, float]:
    """给一个候选分区打无标签分数。

    分数由整体 silhouette、簇大小均衡性和与参考分区的一致性组成。这里的
    reference_labels 来自另一种无监督初始化，不涉及真实设备标签。
    """
    labels = _dense_relabel(labels)
    if len(np.unique(labels)) <= 1:
        silhouette = -1.0
    else:
        silhouette = float(silhouette_score(features, labels, metric="euclidean"))
    balance = 1.0 / (1.0 + _cluster_size_cv(labels))
    stability = float(adjusted_rand_score(reference_labels, labels))
    score = 0.55 * silhouette + 0.25 * stability + 0.20 * balance
    return {
        "score": float(score),
        "silhouette": float(silhouette),
        "balance": float(balance),
        "stability": float(stability),
        "cluster_size_cv": float(_cluster_size_cv(labels)),
    }


def run_gpcc(feats: dict[str, np.ndarray], target_clusters: int, seed: int = 7) -> GPCCResult:
    """运行 GPCC 并返回连续簇标签、样本置信度和诊断。

    参数
    ----
    feats:
        ``build_round_features`` 生成的多视图特征字典，至少包含 ``deep``、
        ``rf`` 和 ``graph``。
    target_clusters:
        协议公开声明的当前轮新增类别数。函数会固定输出该数量的簇。
    seed:
        KMeans 随机种子，用于保证 Slurm 短实验可复现。
    """
    deep = _safe_normalize(feats["deep"])
    rf = _safe_normalize(feats["rf"])
    graph = _safe_normalize(feats["graph"])
    n_samples = int(deep.shape[0])
    k = max(1, min(int(target_clusters), n_samples))
    joint = np.concatenate([0.45 * deep, 0.15 * rf, 0.40 * graph], axis=1).astype(np.float32)

    if k == 1:
        labels = np.zeros(n_samples, dtype=np.int64)
        confidence = np.ones(n_samples, dtype=np.float32)
        return GPCCResult(labels, confidence, {
            "GPCC Backend": "single_cluster_fallback",
            "GPCC Target Clusters": 1,
            "GPCC Selected Candidate": "single",
            "GPCC Silhouette": np.nan,
            "GPCC Cluster Stability": 1.0,
            "GPCC Cluster Size CV": 0.0,
            "GPCC Confidence Mean": 1.0,
            "GPCC Uses HDBSCAN": False,
        })

    km_joint = KMeans(n_clusters=k, n_init=20, random_state=int(seed)).fit_predict(joint)
    km_graph = KMeans(n_clusters=k, n_init=20, random_state=int(seed) + 17).fit_predict(graph)
    agg_joint = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(joint)

    reference = _dense_relabel(km_joint)
    candidates = {
        "kmeans_joint": _dense_relabel(km_joint),
        "kmeans_graph": _dense_relabel(km_graph),
        "agglomerative_joint": _dense_relabel(agg_joint),
    }
    scored = {
        name: _candidate_score(joint, labels, reference)
        for name, labels in candidates.items()
    }
    selected_name = max(scored, key=lambda name: scored[name]["score"])
    labels = candidates[selected_name]

    margin_conf = _prototype_margin_confidence(joint, labels)
    silhouette_conf, silhouette_mean = _silhouette_confidence(joint, labels)
    confidence = np.sqrt(np.clip(margin_conf, 0.0, 1.0) * np.clip(silhouette_conf, 0.0, 1.0))
    confidence = np.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    diagnostics = {
        "GPCC Backend": "graph_prototype_controlled_clustering",
        "GPCC Target Clusters": int(k),
        "GPCC Selected Candidate": selected_name,
        "GPCC Candidate Scores": scored,
        "GPCC Silhouette": float(silhouette_mean),
        "GPCC Prototype Margin Mean": float(np.mean(margin_conf)),
        "GPCC Silhouette Confidence Mean": float(np.mean(silhouette_conf)),
        "GPCC Confidence Mean": float(np.mean(confidence)),
        "GPCC Confidence Min": float(np.min(confidence)),
        "GPCC Cluster Stability": float(scored[selected_name]["stability"]),
        "GPCC Cluster Size CV": float(_cluster_size_cv(labels)),
        "GPCC Uses HDBSCAN": False,
    }
    return GPCCResult(labels.astype(np.int64), confidence, diagnostics)
