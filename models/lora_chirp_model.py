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


class _ComplexGeometryBackbone(nn.Module):
    """从 IQ 输入提取幅度/相位几何信息的轻量分支。

    LoRa 的设备差异不只体现在原始 I/Q 波形，也可能体现在幅度包络、
    相位变化和局部频率轨迹上。该分支只使用当前样本自身可计算的几何量，
    不引入设备标签或评估集信息，作为原始波形分支的互补视图。
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 48, 7, padding=3, bias=False),
            nn.BatchNorm1d(48),
            nn.SiLU(inplace=True),
            nn.Conv1d(48, 96, 9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(96),
            nn.SiLU(inplace=True),
        )
        self.block = _ResidualBlock(96, 160, 7, stride=2, dilation=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(160, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 为 [B, 2, L]；把复数 IQ 转成幅度和相邻采样相位变化。
        i_signal, q_signal = x[:, 0], x[:, 1]
        magnitude = torch.sqrt(i_signal.square() + q_signal.square() + 1e-6)
        phase = torch.atan2(q_signal, i_signal)
        phase_delta = torch.atan2(
            torch.sin(phase[:, 1:] - phase[:, :-1]),
            torch.cos(phase[:, 1:] - phase[:, :-1]),
        )
        phase_delta = F.pad(phase_delta, (1, 0))
        geometry = torch.stack((magnitude, phase_delta), dim=1)
        features = self.stem(geometry)
        features = self.block(features)
        return self.fc(self.pool(features).squeeze(-1))


class LoRaHybridBackbone(nn.Module):
    """原始 Chirp 与复数几何视图的门控多视图 backbone。

    原始 Chirp 分支保留已有的局部/膨胀卷积建模能力；几何分支显式建模
    幅度和相邻相位变化。门控融合让网络在不同天、不同设备上自行选择两
    个视图的贡献，输出维度保持不变，因此不影响后续发现和 CIL 接口。
    """

    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.waveform = LoRaChirpBackbone(feat_dim=feat_dim)
        self.geometry = _ComplexGeometryBackbone(feat_dim=feat_dim)
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        waveform_feat = self.waveform(x)
        geometry_feat = self.geometry(x)
        joined = torch.cat((waveform_feat, geometry_feat), dim=1)
        gate = self.gate(joined)
        fused = torch.cat((waveform_feat * gate, geometry_feat * (1.0 - gate)), dim=1)
        return self.fuse(fused)


class LoRaHybridClosedSet(nn.Module):
    """LoRa 多视图门控 backbone 的闭集分类器。"""

    def __init__(self, num_known_classes: int, feat_dim: int = 128):
        super().__init__()
        self.num_known_classes = num_known_classes
        self.feat_dim = feat_dim
        self.backbone = LoRaHybridBackbone(feat_dim=feat_dim)
        self.classifier = nn.Linear(feat_dim, num_known_classes)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        return feat, self.classifier(feat)
