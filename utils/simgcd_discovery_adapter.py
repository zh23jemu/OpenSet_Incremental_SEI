"""SimGCD 式类别发现前端的最小可运行适配器。

该模块先提供一个严格无泄漏的轻量实现，用于把阶段 1 的
`DiscoveryAdapter` 契约接入真实代码。它不复刻视觉 SimGCD 的完整
训练流程，而是在冻结 RF/IQ embedding 上构造一个参数化余弦发现头：

1. 已知类权重由已知训练特征的类均值初始化；
2. 新类权重由当前轮次未标注特征的 KMeans 簇中心初始化；
3. 预测时只返回未标注样本在新类原型上的操作性簇标签和置信度。

这样做的目的，是先建立 SimGCD/IGCD 后续适配都能复用的接口、诊断
和无真值边界。后续若接入真正的可训练 SimGCD head，可以保持输入
输出不变，只替换 `fit` 中的学习过程。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utils.discovery_adapter_contract import (
    DiscoveryAdapterInput,
    DiscoveryAdapterOutput,
    validate_discovery_input,
    validate_discovery_output,
)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """按行做 L2 归一化，避免余弦相似度受特征尺度影响。"""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def _dense_relabel(labels: np.ndarray) -> np.ndarray:
    """将聚类标签压缩为 0..K-1，保证后续伪标签注册入口稳定。"""
    labels = np.asarray(labels, dtype=np.int64)
    unique = sorted(int(v) for v in np.unique(labels))
    mapping = {old: new for new, old in enumerate(unique)}
    return np.asarray([mapping[int(v)] for v in labels], dtype=np.int64)


@dataclass
class SimGCDStyleDiscoveryAdapter:
    """冻结特征上的 SimGCD 式参数化发现头。

    参数说明：
    - `candidate_k_offsets`：当未显式给出新类数量时，在启发式中心 K
      附近搜索的偏移量。搜索只使用未标注特征的 silhouette，不使用
      未知真实标签。
    - `temperature`：余弦 logits 的 softmax 温度，越小置信度越尖锐。
    - `random_state`：KMeans 随机种子，保证短实验可复现。
    """

    candidate_k_offsets: tuple[int, ...] = (-2, -1, 0, 1, 2)
    temperature: float = 0.1
    random_state: int = 7
    n_init: int = 10
    name: str = "simgcd_style_prototype_head"
    scaler_: StandardScaler | None = field(default=None, init=False)
    known_class_ids_: np.ndarray | None = field(default=None, init=False)
    known_weights_: np.ndarray | None = field(default=None, init=False)
    novel_weights_: np.ndarray | None = field(default=None, init=False)
    estimated_new_classes_: int | None = field(default=None, init=False)
    diagnostics_: dict[str, Any] = field(default_factory=dict, init=False)

    def fit(self, data: DiscoveryAdapterInput) -> None:
        """拟合参数化发现头。

        该方法只读取契约允许的已知训练标签和当前轮次未标注特征。
        未知轮次真实标签、评估集样本和评估集标签不会进入估计过程。
        """
        validate_discovery_input(data)
        known_features = np.asarray(data.known_features, dtype=np.float32)
        known_labels = np.asarray(data.known_labels, dtype=np.int64)
        unlabeled_features = np.asarray(data.unlabeled_features, dtype=np.float32)

        # 标准化仅使用训练侧可见特征：已知训练特征 + 当前 discovery 未标注特征。
        # 这不包含评估集，也不包含未知真值。
        self.scaler_ = StandardScaler()
        joint = np.vstack([known_features, unlabeled_features])
        self.scaler_.fit(joint)
        known_z = _l2_normalize(self.scaler_.transform(known_features).astype(np.float32))
        unlabeled_z = _l2_normalize(self.scaler_.transform(unlabeled_features).astype(np.float32))

        self.known_class_ids_ = np.asarray(sorted(np.unique(known_labels).tolist()), dtype=np.int64)
        known_weights = []
        for class_id in self.known_class_ids_:
            class_features = known_z[known_labels == int(class_id)]
            known_weights.append(class_features.mean(axis=0))
        self.known_weights_ = _l2_normalize(np.vstack(known_weights).astype(np.float32))

        k, k_diagnostics = self._select_new_class_count(unlabeled_z, data.expected_new_classes)
        km = KMeans(n_clusters=k, n_init=int(self.n_init), random_state=int(self.random_state))
        km.fit(unlabeled_z)
        self.novel_weights_ = _l2_normalize(km.cluster_centers_.astype(np.float32))
        self.estimated_new_classes_ = int(k)
        self.diagnostics_ = {
            "adapter": self.name,
            "round_name": data.round_name,
            "known_classes": int(len(self.known_class_ids_)),
            "estimated_new_classes": int(k),
            "temperature": float(self.temperature),
            "class_count_selection": k_diagnostics,
            "uses_unknown_true_labels": False,
            "uses_eval_set": False,
        }

    def predict(self, data: DiscoveryAdapterInput) -> DiscoveryAdapterOutput:
        """返回当前轮次未标注样本的新类簇标签和 label-free 置信度。"""
        validate_discovery_input(data)
        if self.scaler_ is None or self.novel_weights_ is None or self.estimated_new_classes_ is None:
            raise RuntimeError("adapter must be fitted before predict")

        unlabeled_features = np.asarray(data.unlabeled_features, dtype=np.float32)
        unlabeled_z = _l2_normalize(self.scaler_.transform(unlabeled_features).astype(np.float32))
        novel_logits = unlabeled_z @ self.novel_weights_.T
        probs = self._softmax(novel_logits / max(float(self.temperature), 1e-6))
        labels = _dense_relabel(np.argmax(probs, axis=1).astype(np.int64))
        confidence = np.max(probs, axis=1).astype(np.float32)

        output = DiscoveryAdapterOutput(
            cluster_labels=labels,
            confidence=confidence,
            estimated_new_classes=int(self.estimated_new_classes_),
            diagnostics={
                **self.diagnostics_,
                "confidence_mean": float(np.mean(confidence)),
                "confidence_min": float(np.min(confidence)),
                "confidence_max": float(np.max(confidence)),
                "assignment_coverage": 1.0,
            },
        )
        validate_discovery_output(output, expected_samples=unlabeled_features.shape[0])
        return output

    def _select_new_class_count(self, unlabeled_z: np.ndarray, expected_new_classes: int | None) -> tuple[int, dict[str, Any]]:
        """选择新类数量。

        如果协议在实验开始前已经固定 `expected_new_classes`，直接使用该值；
        否则只根据未标注特征的 silhouette 做无标签估计。这个估计不使用
        unknown true labels，因此可以作为后续更强 SimGCD head 的安全占位。
        """
        n_samples = int(unlabeled_z.shape[0])
        if expected_new_classes is not None:
            k = max(1, min(int(expected_new_classes), n_samples))
            return k, {"source": "protocol_expected_new_classes", "selected_k": k}

        heuristic_center = max(2, int(round(np.sqrt(max(n_samples, 2) / 2.0))))
        candidates = sorted(
            {
                k
                for offset in self.candidate_k_offsets
                for k in [heuristic_center + int(offset)]
                if 2 <= k <= min(n_samples - 1, 50)
            }
        )
        if not candidates:
            return 1, {"source": "fallback_single_cluster", "selected_k": 1}

        scored: list[dict[str, float | int]] = []
        for k in candidates:
            km = KMeans(n_clusters=k, n_init=int(self.n_init), random_state=int(self.random_state))
            labels = km.fit_predict(unlabeled_z)
            score = float(silhouette_score(unlabeled_z, labels)) if len(np.unique(labels)) > 1 else -1.0
            scored.append({"k": int(k), "silhouette": score})
        best = max(scored, key=lambda item: float(item["silhouette"]))
        return int(best["k"]), {"source": "unlabeled_silhouette", "selected_k": int(best["k"]), "candidates": scored}

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """稳定 softmax，用于把新类余弦 logits 转为置信度。"""
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)
