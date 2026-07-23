import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    """
    A simple residual 1D convolution block for I/Q signal feature extraction.
    Input shape: [B, C, L]
    Output shape: [B, out_ch, L']
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 1):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv1d(
                in_channels=out_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_ch),
        )

        self.shortcut = nn.Identity()

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        shortcut = self.shortcut(x)
        return F.relu(out + shortcut, inplace=True)


class ResNet1DBackbone(nn.Module):
    """
    A lightweight 1D backbone for SEI / RF fingerprinting.

    Expected input:
        x: [B, 2, L]
           B = batch size
           2 = I/Q channels
           L = signal length

    Output:
        z: [B, feat_dim]
    """

    def __init__(self, feat_dim: int = 128):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = ConvBlock1D(32, 64, kernel_size=7, stride=2)
        self.layer2 = ConvBlock1D(64, 128, kernel_size=5, stride=2)
        self.layer3 = ConvBlock1D(128, 128, kernel_size=3, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 2, L]
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool(x).squeeze(-1)
        z = self.fc(x)

        return z