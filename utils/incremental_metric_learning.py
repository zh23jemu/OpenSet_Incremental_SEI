"""增量阶段的归一化代理度量损失。

该模块只处理训练期的特征几何约束，不参与聚类、验证或最终评估。
增量训练时，分类器权重可以看作每个已注册类别的代理向量；将样本
特征和代理都归一化后再计算交叉熵，可以减少线性分类头的范数偏置，
并让旧类 replay 与高置信新类伪标签共享同一个角度空间。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_proxy_metric_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    classifier_weight: torch.Tensor,
    scale: float = 16.0,
    margin: float = 0.0,
) -> torch.Tensor:
    """计算归一化代理分类损失。

    参数：
        features: 当前训练 batch 的特征，形状为 ``[N, D]``。
        labels: 与分类器输出行对应的连续类别索引，形状为 ``[N]``。
        classifier_weight: 线性分类头权重，形状为 ``[C, D]``。
        scale: 归一化余弦 logits 的缩放系数，必须为正。
        margin: 对真实类别余弦相似度施加的固定角度间隔，范围为
            ``[0, 1)``。默认 0 表示只使用归一化代理损失。

    返回：
        可反向传播的标量损失。输入为空、标签无效或没有有效类别时，
        返回与 ``features`` 位于同一设备和 dtype 的零损失。
    """
    if features.ndim != 2 or classifier_weight.ndim != 2:
        raise ValueError("features 和 classifier_weight 必须是二维张量。")
    if features.shape[1] != classifier_weight.shape[1]:
        raise ValueError("features 与 classifier_weight 的特征维度不一致。")
    if scale <= 0:
        raise ValueError("scale 必须为正数。")
    if margin < 0 or margin >= 1:
        raise ValueError("margin 必须满足 0 <= margin < 1。")
    if features.shape[0] == 0 or labels.numel() == 0:
        return features.sum() * 0.0

    labels = labels.to(device=features.device, dtype=torch.long).reshape(-1)
    if labels.shape[0] != features.shape[0]:
        raise ValueError("labels 数量必须与 features 的样本数一致。")

    valid = (labels >= 0) & (labels < classifier_weight.shape[0])
    if not torch.any(valid):
        return features.sum() * 0.0

    z = F.normalize(features[valid], dim=1)
    proxies = F.normalize(classifier_weight, dim=1)
    cosine_logits = z @ proxies.T

    # 只对真实类别减去 margin，保持负类代理的相对排序不变。
    target = labels[valid]
    if margin > 0:
        row_index = torch.arange(len(target), device=features.device)
        cosine_logits = cosine_logits.clone()
        cosine_logits[row_index, target] -= float(margin)

    return F.cross_entropy(cosine_logits * float(scale), target)

