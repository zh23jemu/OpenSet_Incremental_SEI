import torch
import torch.nn as nn

from models.backbone_1d import ResNet1DBackbone


class ClosedSetSEI(nn.Module):
    """
    Standard closed-set SEI classifier.

    Output:
        logits: [B, K]
        K = number of known transmitters/classes
    """

    def __init__(self, num_known_classes: int, feat_dim: int = 128):
        super().__init__()

        self.num_known_classes = num_known_classes
        self.feat_dim = feat_dim

        self.backbone = ResNet1DBackbone(feat_dim=feat_dim)
        self.classifier = nn.Linear(feat_dim, num_known_classes)

    def forward(self, x: torch.Tensor):
        """
        x: [B, 2, L]

        Returns:
            feat: [B, feat_dim]
            logits: [B, K]
        """
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return feat, logits


class VUPSEI(nn.Module):
    """
    VUP-based SEI classifier.

    VUP = Virtual Unknown Prototype.

    This module follows the PROSER-style placeholder idea:
        K known classifier nodes -> K known nodes + C virtual unknown nodes

    Output:
        logits: [B, K + C]

    The last C logits correspond to VUP / dummy nodes.
    They are not real classes, but unknown reception slots.
    """

    def __init__(
        self,
        num_known_classes: int,
        num_dummy: int = 3,
        feat_dim: int = 128,
    ):
        super().__init__()

        self.num_known_classes = num_known_classes
        self.num_dummy = num_dummy
        self.feat_dim = feat_dim

        self.backbone = ResNet1DBackbone(feat_dim=feat_dim)
        self.classifier = nn.Linear(feat_dim, num_known_classes + num_dummy)

    def forward(self, x: torch.Tensor):
        """
        x: [B, 2, L]

        Returns:
            feat: [B, feat_dim]
            logits: [B, K + C]
        """
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return feat, logits