import torch
import torch.nn.functional as F


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """
    Cross entropy with soft labels.

    logits: [B, num_classes]
    soft_targets: [B, num_classes]
    """
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(soft_targets * log_probs).sum(dim=1).mean()
    return loss


def vup_proser_style_loss(
    features: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    classifier,
    K: int,
    C: int,
    beta: float = 1.0,
    gamma: float = 1.0,
    mix_lam: float = 0.5,
):
    """
    PROSER-style VUP loss for SEI.

    This loss has three parts:

    1. Known classification loss:
       keep known classes correctly classified.

    2. Dummy-second loss:
       remove the true class logit, then encourage dummy/VUP nodes
       to dominate the remaining non-target known classes.
       Intuition:
           True class > Dummy/VUP > Other known classes

    3. Feature-level mixup loss:
       mix features from different known classes and treat the mixed
       features as pseudo-unknown samples.
       These mixed features are trained into dummy/VUP nodes.

    Args:
        features: [B, feat_dim]
        logits: [B, K + C]
        labels: [B], labels must be in [0, K-1]
        classifier: final linear classifier of VUP model
        K: number of known classes
        C: number of VUP/dummy nodes
        beta: weight for dummy-second loss
        gamma: weight for mixup loss
        mix_lam: mixup coefficient

    Returns:
        dict containing total loss and detached sub-losses
    """

    device = features.device
    batch_size = features.size(0)

    if logits.size(1) != K + C:
        raise ValueError(
            f"Expected logits dimension {K + C}, but got {logits.size(1)}"
        )

    if labels.min().item() < 0 or labels.max().item() >= K:
        raise ValueError(
            "Labels for VUP training must be known-class labels in [0, K-1]."
        )

    # ------------------------------------------------------------
    # 1. Known classification loss
    # ------------------------------------------------------------
    # Only use the first K logits for known-class classification.
    loss_known = F.cross_entropy(logits[:, :K], labels)

    # ------------------------------------------------------------
    # 2. Dummy-second loss
    # ------------------------------------------------------------
    # For each known sample, remove its true class logit.
    # Then make dummy/VUP nodes dominate the remaining logits.
    #
    # Example:
    # Original logits:
    #   [Class1, Class2, Class3, VUP1, VUP2, VUP3]
    #
    # If true class is Class1, remove Class1:
    #   [Class2, Class3, VUP1, VUP2, VUP3]
    #
    # Then encourage:
    #   VUP > Class2, Class3
    #
    # Combined with loss_known:
    #   Class1 > VUP > Other known classes

    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[torch.arange(batch_size, device=device), labels] = False

    logits_masked = logits[mask].view(batch_size, K + C - 1)

    # Since labels are from known classes, all C dummy logits are still at the end.
    dummy_logits = logits_masked[:, -C:]

    logsum_all = torch.logsumexp(logits_masked, dim=1)
    logsum_dummy = torch.logsumexp(dummy_logits, dim=1)

    # Maximize dummy probability among masked logits.
    loss_dummy = -(logsum_dummy - logsum_all).mean()

    # ------------------------------------------------------------
    # 3. Feature-level mixup loss
    # ------------------------------------------------------------
    # Mix features from different known classes.
    # Treat the mixed feature as pseudo unknown and train it into VUP nodes.

    perm = torch.randperm(batch_size, device=device)

    feat_i = features
    feat_j = features[perm]

    label_i = labels
    label_j = labels[perm]

    diff_class_mask = label_i != label_j

    loss_mixup = torch.tensor(0.0, device=device)

    if diff_class_mask.sum() > 0:
        feat_mix = (
            mix_lam * feat_i[diff_class_mask]
            + (1.0 - mix_lam) * feat_j[diff_class_mask]
        )

        logits_mix = classifier(feat_mix)

        # Soft target: uniformly assign the pseudo-unknown feature to all VUP nodes.
        soft_targets = torch.zeros_like(logits_mix)
        soft_targets[:, K:K + C] = 1.0 / C

        loss_mixup = soft_cross_entropy(logits_mix, soft_targets)

    # ------------------------------------------------------------
    # Total loss
    # ------------------------------------------------------------
    total_loss = loss_known + beta * loss_dummy + gamma * loss_mixup

    return {
        "loss": total_loss,
        "loss_known": loss_known.detach(),
        "loss_dummy": loss_dummy.detach(),
        "loss_mixup": loss_mixup.detach(),
    }