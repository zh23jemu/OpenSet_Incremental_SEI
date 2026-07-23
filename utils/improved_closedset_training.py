"""Leakage-safe closed-set training helpers for RF fingerprint experiments.

This module keeps the final held-out evaluation split untouched.  It creates a
small stratified validation subset *inside* the Day-1 training portion, uses
that subset for checkpoint selection, applies conservative RF augmentations
only to training samples, and confines supervised contrastive pressure to a
temporary projection head.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


TRAINING_RECIPE_VERSION = "v2_day1_validation_rfaug_projection"


@dataclass(frozen=True)
class RFAugmentConfig:
    """Conservative, label-preserving training-time IQ perturbations."""

    amplitude_min: float = 0.90
    amplitude_max: float = 1.10
    phase_degrees: float = 5.0
    max_time_shift: int = 4
    snr_db_min: float = 30.0
    snr_db_max: float = 40.0


class SupConProjectionHead(nn.Module):
    """Projection space used only while optimizing supervised contrastive loss."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Label-supervised contrastive loss over a batch of projected features."""

    features = F.normalize(features, dim=1)
    labels = labels.contiguous().view(-1, 1)
    batch_size = int(features.shape[0])
    if batch_size <= 1:
        return features.sum() * 0.0

    positive_mask = torch.eq(labels, labels.T).float().to(features.device)
    logits = torch.matmul(features, features.T) / float(temperature)
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()

    self_mask = torch.ones_like(positive_mask) - torch.eye(batch_size, device=features.device)
    positive_mask = positive_mask * self_mask
    exp_logits = torch.exp(logits) * self_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not torch.any(valid):
        return features.sum() * 0.0
    mean_log_prob = (positive_mask * log_prob).sum(dim=1) / (positive_count + 1e-8)
    return -mean_log_prob[valid].mean()


def augment_iq_batch(
    x: torch.Tensor,
    config: Optional[RFAugmentConfig] = None,
) -> torch.Tensor:
    """Apply mild amplitude, phase, time-shift, and AWGN perturbations.

    No carrier-frequency-offset augmentation is used because CFO may itself be
    a transmitter fingerprint.  Evaluation and validation samples never pass
    through this function.
    """

    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"Expected IQ tensor [B, 2, L], got {tuple(x.shape)}")
    cfg = config or RFAugmentConfig()
    out = x.clone()
    batch_size = out.shape[0]

    amplitude = torch.empty(batch_size, 1, 1, device=out.device, dtype=out.dtype).uniform_(
        cfg.amplitude_min, cfg.amplitude_max
    )
    out = out * amplitude

    max_phase = float(np.deg2rad(cfg.phase_degrees))
    phase = torch.empty(batch_size, 1, device=out.device, dtype=out.dtype).uniform_(
        -max_phase, max_phase
    )
    cos_phase = torch.cos(phase)
    sin_phase = torch.sin(phase)
    i_data = out[:, 0, :].clone()
    q_data = out[:, 1, :].clone()
    out[:, 0, :] = i_data * cos_phase - q_data * sin_phase
    out[:, 1, :] = i_data * sin_phase + q_data * cos_phase

    if cfg.max_time_shift > 0:
        shifts = torch.randint(
            -cfg.max_time_shift,
            cfg.max_time_shift + 1,
            (batch_size,),
            device=out.device,
        )
        out = torch.stack(
            [torch.roll(out[index], shifts=int(shifts[index].item()), dims=-1) for index in range(batch_size)],
            dim=0,
        )

    signal_power = out.square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    snr_db = torch.empty(batch_size, 1, 1, device=out.device, dtype=out.dtype).uniform_(
        cfg.snr_db_min, cfg.snr_db_max
    )
    noise_power = signal_power / torch.pow(10.0, snr_db / 10.0)
    out = out + torch.randn_like(out) * torch.sqrt(noise_power)
    return out


def _labels_from_dataset(dataset: Dataset) -> np.ndarray:
    if not isinstance(dataset, TensorDataset) or len(dataset.tensors) < 2:
        raise TypeError("Day-1 training data must be a TensorDataset(X, y)")
    return dataset.tensors[1].detach().cpu().numpy().astype(np.int64, copy=False)


def stratified_train_validation_split(
    dataset: Dataset,
    validation_fraction: float,
    seed: int,
    groups: Optional[np.ndarray] = None,
) -> Tuple[Subset, Subset]:
    """Create a deterministic class-stratified Day-1 split.

    When recording/session identifiers are available, ``groups`` keeps every
    recording wholly on one side of the split.
    """

    if not (0.0 < validation_fraction < 0.5):
        raise ValueError("validation_fraction must be in (0, 0.5)")
    labels = _labels_from_dataset(dataset)
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(labels):
            raise ValueError("groups must have the same length as the Day-1 dataset")
    rng = np.random.default_rng(int(seed))
    train_indices = []
    validation_indices = []

    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        if len(class_indices) < 2:
            raise ValueError(f"Class {class_id} has fewer than two Day-1 samples")
        if groups is None:
            class_indices = rng.permutation(class_indices)
            validation_count = max(1, int(round(len(class_indices) * validation_fraction)))
            validation_count = min(validation_count, len(class_indices) - 1)
            validation_indices.extend(class_indices[:validation_count].tolist())
            train_indices.extend(class_indices[validation_count:].tolist())
        else:
            class_groups = np.unique(groups[class_indices])
            if len(class_groups) < 2:
                raise ValueError(
                    f"Class {class_id} has fewer than two recording groups; "
                    "group-disjoint validation is impossible"
                )
            class_groups = rng.permutation(class_groups)
            validation_group_count = max(1, int(round(len(class_groups) * validation_fraction)))
            validation_group_count = min(validation_group_count, len(class_groups) - 1)
            validation_groups = set(class_groups[:validation_group_count].tolist())
            for index in class_indices:
                if groups[index] in validation_groups:
                    validation_indices.append(int(index))
                else:
                    train_indices.append(int(index))

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


@torch.no_grad()
def evaluate_classifier(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        _, logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        loss_sum += float(loss.detach().cpu())
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1), loss_sum / max(total, 1)


def train_closedset_with_validation(
    *,
    train_set: Dataset,
    model_factory: Callable[[], nn.Module],
    num_classes: int,
    feat_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    save_path: str,
    seed: int,
    use_supcon: bool,
    supcon_weight: float,
    supcon_temperature: float,
    checkpoint_metadata: Optional[dict],
    validation_fraction: float = 1.0 / 7.0,
    use_rf_augmentation: bool = True,
    projection_hidden_dim: int = 128,
    projection_dim: int = 64,
    augmentation_config: Optional[RFAugmentConfig] = None,
    validation_groups: Optional[np.ndarray] = None,
) -> None:
    """Train and save the checkpoint with the best Day-1 validation accuracy."""

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    train_subset, validation_subset = stratified_train_validation_split(
        train_set,
        validation_fraction=validation_fraction,
        seed=seed,
        groups=validation_groups,
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=max(batch_size, 256),
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model = model_factory().to(device)
    projector = None
    parameters = list(model.parameters())
    if use_supcon:
        projector = SupConProjectionHead(
            input_dim=feat_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_dim,
        ).to(device)
        parameters.extend(projector.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
    cfg = augmentation_config or RFAugmentConfig()

    print(
        f"\n[Train-v2] classes={num_classes} | train={len(train_subset)} | "
        f"validation={len(validation_subset)} | validation_fraction={validation_fraction:.6f}"
    )
    print(
        f"[Train-v2] RF augmentation={use_rf_augmentation} | "
        f"SupCon projection={projection_dim if use_supcon else 'disabled'}"
    )

    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        if projector is not None:
            projector.train()
        total = 0
        correct = 0
        loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            view_one = augment_iq_batch(x, cfg) if use_rf_augmentation else x
            feat_one, logits_one = model(view_one)
            ce_loss = F.cross_entropy(logits_one, y)
            loss = ce_loss

            if projector is not None:
                view_two = augment_iq_batch(x, cfg) if use_rf_augmentation else x
                feat_two, _ = model(view_two)
                projected = torch.cat([projector(feat_one), projector(feat_two)], dim=0)
                repeated_labels = torch.cat([y, y], dim=0)
                contrastive_loss = supervised_contrastive_loss(
                    projected,
                    repeated_labels,
                    temperature=supcon_temperature,
                )
                loss = ce_loss + float(supcon_weight) * contrastive_loss

            loss.backward()
            optimizer.step()
            correct += int((logits_one.argmax(dim=1) == y).sum().item())
            total += int(y.numel())
            loss_sum += float(loss.detach().cpu())

        train_accuracy = correct / max(total, 1)
        validation_accuracy, validation_loss = evaluate_classifier(model, validation_loader, device)
        improved = validation_accuracy > best_validation_accuracy + 1e-12
        tied_but_lower_loss = (
            abs(validation_accuracy - best_validation_accuracy) <= 1e-12
            and validation_loss < best_validation_loss
        )
        if improved or tied_but_lower_loss:
            best_validation_accuracy = validation_accuracy
            best_validation_loss = validation_loss
            metadata = dict(checkpoint_metadata or {})
            metadata.update(
                {
                    "training_recipe_version": TRAINING_RECIPE_VERSION,
                    "closedset_validation_fraction": float(validation_fraction),
                    "group_disjoint_validation": bool(validation_groups is not None),
                    "rf_augmentation": bool(use_rf_augmentation),
                    "supcon_projection_dim": int(projection_dim) if use_supcon else 0,
                    "supcon_projection_hidden_dim": int(projection_hidden_dim) if use_supcon else 0,
                    "rf_augmentation_config": asdict(cfg) if use_rf_augmentation else None,
                }
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_known_classes": int(num_classes),
                    "feat_dim": int(feat_dim),
                    "metadata": metadata,
                    "selection": {
                        "epoch": int(epoch),
                        "validation_accuracy": float(validation_accuracy),
                        "validation_loss": float(validation_loss),
                    },
                },
                save_path,
            )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={loss_sum / max(len(train_loader), 1):.4f} | "
            f"train_acc={train_accuracy:.4f} | val_acc={validation_accuracy:.4f} | "
            f"val_loss={validation_loss:.4f}"
        )

    print(
        f"[Train-v2] best_val_acc={best_validation_accuracy:.4f} | "
        f"best_val_loss={best_validation_loss:.4f} | saved={save_path}"
    )
