"""LoRa chirp-friendly 1-D backbone for short aligned LoRa SF7 symbols.

该 backbone 针对紧凑 NPZ 中 256 长度的 LoRa SF7 chirp 边界对齐片段设计：

1. 多尺度 stem：7/15/31 三种卷积核并行，覆盖 chirp 斜率与瞬时频率结构；
2. dilation 递增的残差块：增大感受野而不快速缩短序列，避免 256 长度被过早压成 1;
3. attention pooling：在不同 chirp 阶段自适应加权，比单纯 mean pooling 更稳;
4. 输出仍为 ``feat_dim``，``LoRaChirpClosedSet`` 保持与 ``ClosedSetSEI`` 一致的
   ``(feat, logits)`` 接口，方便复用 RADCIL 扩头、SSL 适配和评估链路。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualBlock(nn.Module):
    """残差块，支持 dilation 以扩大感受野。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        # 同尺寸 padding 保持序列长度，dilation 用于扩大感受野。
        padding = dilation * (kernel_size // 2)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        # 通道数或空间分辨率不一致时使用 1x1 投影 shortcut。
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.silu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.silu(x + residual, inplace=True)


class LoRaChirpBackbone(nn.Module):
    """LoRa 短 chirp 序列的多尺度 dilation backbone。"""

    def __init__(self, feat_dim: int = 128):
        super().__init__()
        # 三种卷积核并行，覆盖 LoRa chirp 不同时间尺度的瞬态结构。
        self.stem_7 = nn.Conv1d(2, 32, 7, stride=1, padding=3, bias=False)
        self.stem_15 = nn.Conv1d(2, 32, 15, stride=1, padding=7, bias=False)
        self.stem_31 = nn.Conv1d(2, 32, 31, stride=1, padding=15, bias=False)
        self.stem_bn = nn.BatchNorm1d(96)
        # 第一层 stride=2 开始降采样，后续保持长度，让 dilation 有意义。
        self.layer1 = _ResidualBlock(96, 128, 9, stride=2, dilation=1)
        self.layer2 = _ResidualBlock(128, 192, 7, stride=2, dilation=2)
        self.layer3 = _ResidualBlock(192, 256, 5, stride=1, dilation=4)
        # attention pooling 让模型自适应挑选 chirp 中关键时间片段。
        self.attention = nn.Sequential(
            nn.Conv1d(256, 64, 1),
            nn.Tanh(),
            nn.Conv1d(64, 1, 1),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, L]，其中 L=256 来自紧凑 NPZ 的对齐片段。
        x = torch.cat([self.stem_7(x), self.stem_15(x), self.stem_31(x)], dim=1)
        x = F.silu(self.stem_bn(x), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        weights = torch.softmax(self.attention(x), dim=-1)
        pooled = torch.sum(x * weights, dim=-1)
        return self.fc(pooled)


class LoRaChirpClosedSet(nn.Module):
    """LoRa chirp-friendly closed-set classifier，与 ClosedSetSEI 接口一致。"""

    def __init__(self, num_known_classes: int, feat_dim: int = 128):
        super().__init__()
        self.num_known_classes = num_known_classes
        self.feat_dim = feat_dim
        self.backbone = LoRaChirpBackbone(feat_dim=feat_dim)
        self.classifier = nn.Linear(feat_dim, num_known_classes)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        return feat, self.classifier(feat)
