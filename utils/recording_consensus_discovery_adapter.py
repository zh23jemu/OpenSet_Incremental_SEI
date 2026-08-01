"""Recording-level GPCC 发现适配器。

LoRa Different Days Indoor 子集里，一个 ``recording_id`` 对应同一设备、
同一天、同一次 IQ transmission 下切出的多段 aligned symbol。这些
symbol 共享可观测的物理录制边界，但未知设备真值不能用于训练或聚类。

本模块只利用 ``recording_id`` 这个无标签元数据做 must-link 聚合：
先把同一 recording 的 deep/RF/graph 特征平均成组级原型，再运行现有
GPCC 固定 K 聚类，最后把组标签和置信度回填给每个 symbol。这样可以
减少单段 symbol 噪声对 LoRa 新类注册的影响，同时保持 strict 协议：
不读取 held-out eval，不读取未知真实标签，也不调用 HDBSCAN。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.graph_prototype_discovery_adapter import GPCCResult, run_gpcc


@dataclass(frozen=True)
class RecordingGPCCResult:
    """Recording 级 GPCC 的样本级输出。

    ``labels`` 和 ``confidence`` 始终与 symbol 样本一一对应，保证下游
    RADCIL、回放记忆和现有 CSV 报告不需要理解 recording 级中间结构。
    """

    labels: np.ndarray
    confidence: np.ndarray
    diagnostics: dict[str, Any]


def run_recording_consensus_gpcc(
    feats: dict[str, np.ndarray],
    recording_ids: np.ndarray,
    target_clusters: int,
    seed: int = 7,
    consensus_threshold: float = 0.75,
) -> RecordingGPCCResult:
    """先做 symbol 级 GPCC，再对高一致 recording 做软 must-link 修正。

    与 ``run_recording_gpcc`` 的整段 recording 平均不同，这里保留每个 symbol
    的原始多视图特征，避免一个 transmission 内的幅度/相位变化被平均掉。只有
    当一个 recording 内至少达到 ``consensus_threshold`` 的样本已经属于同一簇
    时，才把该组样本统一到多数簇；低一致 recording 不强行合并，避免把跨设备
    或跨 symbol 的混合结构错误压成单一伪类。该过程只使用可观测 recording_id
    和当前 discovery 的无标签聚类结果，不读取 held-out 真值。
    """
    recording_ids = np.asarray(recording_ids)
    sample_count = int(np.asarray(feats["deep"]).shape[0])
    if recording_ids.shape != (sample_count,):
        raise ValueError(
            f"recording_ids must have shape ({sample_count},), got {recording_ids.shape}"
        )
    threshold = float(consensus_threshold)
    if not 0.5 <= threshold <= 1.0:
        raise ValueError("consensus_threshold must be within [0.5, 1.0].")

    base: GPCCResult = run_gpcc(feats, target_clusters=int(target_clusters), seed=int(seed))
    labels = np.asarray(base.labels, dtype=np.int64).copy()
    confidence = np.asarray(base.confidence, dtype=np.float32).copy()
    unique_recordings, inverse = np.unique(recording_ids, return_inverse=True)
    changed = np.zeros(sample_count, dtype=bool)
    accepted_groups = 0
    group_consistency = []
    for group_id in range(len(unique_recordings)):
        idx = np.where(inverse == group_id)[0]
        if len(idx) == 0:
            continue
        values, counts = np.unique(labels[idx], return_counts=True)
        majority_index = int(np.argmax(counts))
        majority_label = int(values[majority_index])
        consistency = float(counts[majority_index] / len(idx))
        group_consistency.append(consistency)
        if consistency >= threshold and len(values) > 1:
            changed[idx] = labels[idx] != majority_label
            labels[idx] = majority_label
            confidence[idx] = np.maximum(confidence[idx], consistency).astype(np.float32)
            accepted_groups += 1

    diagnostics = dict(base.diagnostics)
    diagnostics.update({
        "Recording-Consensus Backend": "symbol_gpcc_with_selective_recording_consensus",
        "Recording-Consensus Groups": int(len(unique_recordings)),
        "Recording-Consensus Threshold": threshold,
        "Recording-Consensus Accepted Groups": int(accepted_groups),
        "Recording-Consensus Changed Samples": int(np.sum(changed)),
        "Recording-Consensus Mean Consistency": float(np.mean(group_consistency)) if group_consistency else 0.0,
        "Recording-Consensus Min Consistency": float(np.min(group_consistency)) if group_consistency else 0.0,
    })
    return RecordingGPCCResult(labels=labels, confidence=confidence, diagnostics=diagnostics)


def _aggregate_view_by_recording(view: np.ndarray, inverse: np.ndarray, group_count: int) -> np.ndarray:
    """按 recording 组平均单一路特征。

    参数
    ----
    view:
        样本级特征，形状为 ``[N, D]``。
    inverse:
        ``np.unique(recording_ids, return_inverse=True)`` 返回的样本到组索引。
    group_count:
        当前 discovery 轮里的 recording 组数量。
    """

    view = np.asarray(view, dtype=np.float32)
    grouped = np.zeros((int(group_count), view.shape[1]), dtype=np.float32)
    counts = np.bincount(inverse, minlength=int(group_count)).astype(np.float32)
    for feature_dim in range(view.shape[1]):
        grouped[:, feature_dim] = np.bincount(
            inverse,
            weights=view[:, feature_dim],
            minlength=int(group_count),
        )
    grouped /= np.maximum(counts[:, None], 1.0)
    return grouped.astype(np.float32)


def _group_compactness_confidence(features: np.ndarray, inverse: np.ndarray, group_count: int) -> np.ndarray:
    """估计每个样本在 recording 内的一致性置信度。

    同一 recording 内的 symbol 如果围绕组中心比较紧，说明这次录制本身
    稳定；如果离组中心很远，则降低一点样本置信度。这个置信度不参与
    聚类选参，只作为 RADCIL 伪标签权重的辅助输入。
    """

    z = np.asarray(features, dtype=np.float32)
    centers = _aggregate_view_by_recording(z, inverse, group_count)
    distances = np.linalg.norm(z - centers[inverse], axis=1)
    if distances.size == 0:
        return np.ones(0, dtype=np.float32)
    scale = float(np.median(distances) + 1e-8)
    compactness = np.exp(-distances / scale)
    return np.clip(compactness, 0.05, 1.0).astype(np.float32)


def run_recording_gpcc(
    feats: dict[str, np.ndarray],
    recording_ids: np.ndarray,
    target_clusters: int,
    seed: int = 7,
) -> RecordingGPCCResult:
    """运行 recording 级 GPCC，并返回样本级标签。

    ``recording_ids`` 是 LoRa loader 提供的可观测录制编号。函数先在组级
    原型上聚类，再将聚类结果展开到原始 symbol 样本，因此下游 RADCIL、
    replay memory 和已有报告 CSV 都不需要理解组级结构。
    """

    recording_ids = np.asarray(recording_ids)
    sample_count = int(np.asarray(feats["deep"]).shape[0])
    if recording_ids.shape != (sample_count,):
        raise ValueError(
            f"recording_ids must have shape ({sample_count},), got {recording_ids.shape}"
        )
    unique_recordings, inverse = np.unique(recording_ids, return_inverse=True)
    group_count = int(unique_recordings.size)
    if group_count < int(target_clusters):
        raise ValueError(
            "recording-level GPCC requires at least target_clusters recording groups: "
            f"groups={group_count}, target={int(target_clusters)}"
        )

    group_feats = {
        key: _aggregate_view_by_recording(feats[key], inverse, group_count)
        for key in ("deep", "rf", "graph")
    }
    group_result: GPCCResult = run_gpcc(
        group_feats,
        target_clusters=int(target_clusters),
        seed=int(seed),
    )
    labels = group_result.labels[inverse].astype(np.int64)
    group_confidence = group_result.confidence[inverse].astype(np.float32)
    compactness = _group_compactness_confidence(feats["graph"], inverse, group_count)
    confidence = np.sqrt(np.clip(group_confidence, 0.0, 1.0) * compactness)
    confidence = np.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    _, group_sizes = np.unique(inverse, return_counts=True)
    diagnostics = {
        "Recording-GPCC Backend": "recording_level_gpcc",
        "Recording-GPCC Groups": int(group_count),
        "Recording-GPCC Target Clusters": int(target_clusters),
        "Recording-GPCC Mean Group Size": float(np.mean(group_sizes)),
        "Recording-GPCC Min Group Size": int(np.min(group_sizes)),
        "Recording-GPCC Max Group Size": int(np.max(group_sizes)),
        "Recording-GPCC Confidence Mean": float(np.mean(confidence)),
        "Recording-GPCC Compactness Mean": float(np.mean(compactness)),
        **{
            f"Group {key}": value
            for key, value in group_result.diagnostics.items()
        },
    }
    return RecordingGPCCResult(labels=labels, confidence=confidence, diagnostics=diagnostics)
