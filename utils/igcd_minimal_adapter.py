"""IGCD 严格协议最小发现入口。

该模块提供一个可运行的 IGCD-style baseline 骨架，用于阶段 1 先验证
“Incremental Generalized Category Discovery”能否按本项目 60/10/30
strict 协议接入。它不是视觉 IGCD 论文的完整复现，而是一个安全的
最小入口：

1. 只接收已知类训练/验证 embedding 与当前轮 discovery 未标注 embedding；
2. 使用已知类原型作为旧类锚点；
3. 使用当前轮未标注 embedding 的无标签 KMeans 原型作为候选新类；
4. 输出当前轮新类簇标签和 label-free 置信度，供后续报告或 RADCIL
   伪标签入口复用。

后续若接入完整 IGCD 训练、采样或非参数分类逻辑，应保持本模块的
输入输出边界不变，避免把未知真值或 held-out eval 集引入适配过程。
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
    """按行做 L2 归一化，保证后续距离和余弦计算的尺度稳定。"""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def _dense_relabel(labels: np.ndarray) -> np.ndarray:
    """把任意聚类标签压缩为连续 0..K-1，便于后续伪标签注册。"""
    labels = np.asarray(labels, dtype=np.int64)
    unique = sorted(int(v) for v in np.unique(labels))
    mapping = {old: new for new, old in enumerate(unique)}
    return np.asarray([mapping[int(v)] for v in labels], dtype=np.int64)


def _softplus_margin_confidence(margin: np.ndarray) -> np.ndarray:
    """将新类原型相对旧类原型的距离优势转为 [0, 1] 置信度。

    margin 越大，表示样本离新类原型更近、离旧类原型更远，越像可靠
    新类样本。这里不使用真实标签，只基于适配器可见 embedding。
    """
    margin = np.asarray(margin, dtype=np.float32)
    scale = float(np.std(margin) + 1e-6)
    centered = (margin - float(np.median(margin))) / scale
    return (1.0 / (1.0 + np.exp(-centered))).astype(np.float32)


@dataclass
class IGCDMinimalDiscoveryAdapter:
    """IGCD-style 多时间步发现前端的最小安全实现。

    参数说明：
    - `candidate_k_offsets`：未显式给出新类数量时，在启发式中心附近
      做无标签 silhouette 搜索；
    - `random_state`：KMeans 随机种子，保证短实验可复现；
    - `n_init`：KMeans 初始化次数；
    - `name`：写入报告的适配器名称。
    """

    candidate_k_offsets: tuple[int, ...] = (-2, -1, 0, 1, 2)
    random_state: int = 7
    n_init: int = 10
    name: str = "igcd_minimal_prototype_discovery"
    scaler_: StandardScaler | None = field(default=None, init=False)
    known_class_ids_: np.ndarray | None = field(default=None, init=False)
    known_prototypes_: np.ndarray | None = field(default=None, init=False)
    novel_prototypes_: np.ndarray | None = field(default=None, init=False)
    estimated_new_classes_: int | None = field(default=None, init=False)
    diagnostics_: dict[str, Any] = field(default_factory=dict, init=False)

    def fit(self, data: DiscoveryAdapterInput) -> None:
        """拟合当前时间步的 IGCD-style 原型发现状态。

        该方法不会读取未知真值。`validation_features` 只作为允许存在的
        strict 协议输入边界记录，不参与当前最小实现的类别数搜索。
        """
        validate_discovery_input(data)
        known_features = np.asarray(data.known_features, dtype=np.float32)
        known_labels = np.asarray(data.known_labels, dtype=np.int64)
        unlabeled_features = np.asarray(data.unlabeled_features, dtype=np.float32)

        # IGCD 的严格入口需要同时看到旧类锚点和当前时间步未标注样本。
        # 标准化只使用这两部分可见数据，不触碰 held-out eval。
        self.scaler_ = StandardScaler()
        joint = np.vstack([known_features, unlabeled_features])
        self.scaler_.fit(joint)
        known_z = _l2_normalize(self.scaler_.transform(known_features).astype(np.float32))
        unlabeled_z = _l2_normalize(self.scaler_.transform(unlabeled_features).astype(np.float32))

        self.known_class_ids_ = np.asarray(sorted(np.unique(known_labels).tolist()), dtype=np.int64)
        known_prototypes = []
        for class_id in self.known_class_ids_:
            class_features = known_z[known_labels == int(class_id)]
            known_prototypes.append(class_features.mean(axis=0))
        self.known_prototypes_ = _l2_normalize(np.vstack(known_prototypes).astype(np.float32))

        k, k_diagnostics = self._select_new_class_count(unlabeled_z, data.expected_new_classes)
        km = KMeans(n_clusters=k, n_init=int(self.n_init), random_state=int(self.random_state))
        km.fit(unlabeled_z)
        self.novel_prototypes_ = _l2_normalize(km.cluster_centers_.astype(np.float32))
        self.estimated_new_classes_ = int(k)
        self.diagnostics_ = {
            "adapter": self.name,
            "round_name": data.round_name,
            "protocol_mapping": "IGCD time step -> WiSig strict incremental round",
            "known_classes": int(len(self.known_class_ids_)),
            "estimated_new_classes": int(k),
            "class_count_selection": k_diagnostics,
            "uses_unknown_true_labels": False,
            "uses_eval_set": False,
            "implementation_level": "minimal_strict_entry_not_full_paper_reproduction",
        }

    def predict(self, data: DiscoveryAdapterInput) -> DiscoveryAdapterOutput:
        """返回当前时间步未标注样本的新类簇标签和可靠性置信度。"""
        validate_discovery_input(data)
        if (
            self.scaler_ is None
            or self.known_prototypes_ is None
            or self.novel_prototypes_ is None
            or self.estimated_new_classes_ is None
        ):
            raise RuntimeError("adapter must be fitted before predict")

        unlabeled_features = np.asarray(data.unlabeled_features, dtype=np.float32)
        unlabeled_z = _l2_normalize(self.scaler_.transform(unlabeled_features).astype(np.float32))

        # 使用距离最近的新类原型生成操作性簇标签。
        novel_dist = self._squared_distance(unlabeled_z, self.novel_prototypes_)
        known_dist = self._squared_distance(unlabeled_z, self.known_prototypes_)
        raw_labels = np.argmin(novel_dist, axis=1).astype(np.int64)
        labels = _dense_relabel(raw_labels)

        assigned_novel_dist = np.min(novel_dist, axis=1)
        nearest_known_dist = np.min(known_dist, axis=1)
        margin = nearest_known_dist - assigned_novel_dist
        confidence = _softplus_margin_confidence(margin)

        output = DiscoveryAdapterOutput(
            cluster_labels=labels,
            confidence=confidence,
            estimated_new_classes=int(self.estimated_new_classes_),
            diagnostics={
                **self.diagnostics_,
                "assignment_coverage": 1.0,
                "confidence_mean": float(np.mean(confidence)),
                "confidence_min": float(np.min(confidence)),
                "confidence_max": float(np.max(confidence)),
                "novel_margin_mean": float(np.mean(margin)),
                "novel_margin_min": float(np.min(margin)),
            },
        )
        validate_discovery_output(output, expected_samples=unlabeled_features.shape[0])
        return output

    def _select_new_class_count(self, unlabeled_z: np.ndarray, expected_new_classes: int | None) -> tuple[int, dict[str, Any]]:
        """选择当前时间步新类数量。

        协议若已预先固定每轮新类数，则直接使用该值；否则只用未标注
        embedding 的 silhouette 做无标签估计。
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
    def _squared_distance(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """计算样本到原型的平方欧氏距离矩阵。"""
        diff = x[:, None, :] - centers[None, :, :]
        return np.sum(diff * diff, axis=2)
