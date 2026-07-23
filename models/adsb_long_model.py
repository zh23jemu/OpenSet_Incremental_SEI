"""Long-sequence 1-D network for 4800-sample ADS-B IQ records."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding,
            dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.silu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.silu(x + residual, inplace=True)


class ADSBLongBackbone(nn.Module):
    """Multi-scale front-end plus attention pooling for long RF sequences."""

    def __init__(self, feat_dim=128):
        super().__init__()
        self.stem_7 = nn.Conv1d(2, 32, 7, stride=2, padding=3, bias=False)
        self.stem_15 = nn.Conv1d(2, 32, 15, stride=2, padding=7, bias=False)
        self.stem_31 = nn.Conv1d(2, 32, 31, stride=2, padding=15, bias=False)
        self.stem_bn = nn.BatchNorm1d(96)
        self.stem_pool = nn.MaxPool1d(4, stride=4)
        self.layer1 = ResidualBlock(96, 128, 9, stride=2)
        self.layer2 = ResidualBlock(128, 256, 7, stride=2, dilation=2)
        self.layer3 = ResidualBlock(256, 256, 5, stride=2, dilation=2)
        self.attention = nn.Sequential(
            nn.Conv1d(256, 64, 1),
            nn.Tanh(),
            nn.Conv1d(64, 1, 1),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def forward(self, x):
        x = torch.cat([self.stem_7(x), self.stem_15(x), self.stem_31(x)], dim=1)
        x = self.stem_pool(F.silu(self.stem_bn(x), inplace=True))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        weights = torch.softmax(self.attention(x), dim=-1)
        pooled = torch.sum(x * weights, dim=-1)
        return self.fc(pooled)


class ADSBLongClosedSet(nn.Module):
    def __init__(self, num_known_classes=90, feat_dim=128):
        super().__init__()
        self.num_known_classes = num_known_classes
        self.feat_dim = feat_dim
        self.backbone = ADSBLongBackbone(feat_dim=feat_dim)
        self.classifier = nn.Linear(feat_dim, num_known_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return feat, self.classifier(feat)
