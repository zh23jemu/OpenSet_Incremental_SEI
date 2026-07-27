"""开放集类别发现适配器的最小接口契约。

该模块服务于阶段 1 的 SimGCD / IGCD 适配工作。它不实现具体算法，
只定义统一输入、输出和无泄漏校验，确保后续学习式发现头接入时
不会绕过 `stage1_discovery_adapter_contract.json` 里约定的边界。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class DiscoveryAdapterInput:
    """发现适配器输入。

    只允许包含已知类训练/验证特征和当前轮次未标注特征。未知轮次
    真实标签、评估集特征和评估集标签都不应出现在该对象中。
    """

    known_features: np.ndarray
    known_labels: np.ndarray
    unlabeled_features: np.ndarray
    validation_features: np.ndarray | None = None
    validation_labels: np.ndarray | None = None
    round_name: str = ""
    expected_new_classes: int | None = None


@dataclass(frozen=True)
class DiscoveryAdapterOutput:
    """发现适配器输出。

    `cluster_labels` 是当前轮次未标注样本的操作性簇标签，不要求等同
    于真实设备 ID；`confidence` 是无真值置信度，供后续伪标签筛选或
    加权训练使用。
    """

    cluster_labels: np.ndarray
    confidence: np.ndarray
    estimated_new_classes: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


class DiscoveryAdapter(Protocol):
    """SimGCD、IGCD 或其它发现前端应实现的最小协议。"""

    name: str

    def fit(self, data: DiscoveryAdapterInput) -> None:
        """只基于允许输入拟合发现头或校准规则。"""

    def predict(self, data: DiscoveryAdapterInput) -> DiscoveryAdapterOutput:
        """返回当前轮次未标注样本的簇标签与置信度。"""


def validate_discovery_input(data: DiscoveryAdapterInput) -> None:
    """校验适配器输入形状，提前阻断明显的数据边界错误。

    该函数无法自动判断调用者是否偷偷传入了未知真值，但它会检查
    数组长度、维度和验证特征/标签配对，作为实现 SimGCD/IGCD 适配
    时的第一层保护。
    """
    known_features = np.asarray(data.known_features)
    known_labels = np.asarray(data.known_labels)
    unlabeled_features = np.asarray(data.unlabeled_features)

    if known_features.ndim != 2:
        raise ValueError(f"known_features must be [N, D], got {known_features.shape}")
    if unlabeled_features.ndim != 2:
        raise ValueError(f"unlabeled_features must be [N, D], got {unlabeled_features.shape}")
    if known_labels.ndim != 1:
        raise ValueError(f"known_labels must be [N], got {known_labels.shape}")
    if known_features.shape[0] != known_labels.shape[0]:
        raise ValueError("known_features and known_labels must have the same sample count")
    if known_features.shape[1] != unlabeled_features.shape[1]:
        raise ValueError("known_features and unlabeled_features must share feature dimension")

    has_val_features = data.validation_features is not None
    has_val_labels = data.validation_labels is not None
    if has_val_features != has_val_labels:
        raise ValueError("validation_features and validation_labels must be provided together")
    if has_val_features:
        validation_features = np.asarray(data.validation_features)
        validation_labels = np.asarray(data.validation_labels)
        if validation_features.ndim != 2:
            raise ValueError(f"validation_features must be [N, D], got {validation_features.shape}")
        if validation_labels.ndim != 1:
            raise ValueError(f"validation_labels must be [N], got {validation_labels.shape}")
        if validation_features.shape[0] != validation_labels.shape[0]:
            raise ValueError("validation_features and validation_labels must have the same sample count")
        if validation_features.shape[1] != known_features.shape[1]:
            raise ValueError("validation_features must share feature dimension with known_features")


def validate_discovery_output(output: DiscoveryAdapterOutput, expected_samples: int) -> None:
    """校验发现输出是否能安全交给后续伪标签和增量训练流程。"""
    cluster_labels = np.asarray(output.cluster_labels)
    confidence = np.asarray(output.confidence)

    if cluster_labels.ndim != 1:
        raise ValueError(f"cluster_labels must be [N], got {cluster_labels.shape}")
    if confidence.ndim != 1:
        raise ValueError(f"confidence must be [N], got {confidence.shape}")
    if cluster_labels.shape[0] != expected_samples:
        raise ValueError("cluster_labels length must match unlabeled sample count")
    if confidence.shape[0] != expected_samples:
        raise ValueError("confidence length must match unlabeled sample count")
    if output.estimated_new_classes < 0:
        raise ValueError("estimated_new_classes must be non-negative")
    if np.any(~np.isfinite(confidence)):
        raise ValueError("confidence must contain only finite values")
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("confidence must be in [0, 1]")
