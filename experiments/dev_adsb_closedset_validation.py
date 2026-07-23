"""ADS-B 90-class closed-set development without test-set access.

Only X_train_90Class.npy/Y_train_90Class.npy are opened.  Within each class,
stored-order blocks allocate 60% to training, the next 10% to validation, and
leave the final 30% untouched.  Model selection is validation-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.adsb_90known_strict_loader import _as_int_labels, _load_npy, _take_and_convert
from models.vup_model import ClosedSetSEI
from models.adsb_long_model import ADSBLongClosedSet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_train_validation(data_root: str):
    root = Path(data_root).resolve()
    x_mmap = _load_npy(root, "X_train_90Class.npy")
    y = _as_int_labels(_load_npy(root, "Y_train_90Class.npy"))
    if not np.array_equal(np.unique(y), np.arange(90)):
        raise ValueError("Expected exactly labels 0..89 in the base file.")

    train_indices, val_indices = [], []
    audit = []
    for class_id in range(90):
        indices = np.flatnonzero(y == class_id)
        n = len(indices)
        train_end = int(np.floor(0.60 * n))
        val_end = int(np.floor(0.70 * n))
        if train_end < 1 or val_end <= train_end or val_end >= n:
            raise ValueError(f"Class {class_id} is too small for 60/10/30.")
        train_indices.append(indices[:train_end])
        val_indices.append(indices[train_end:val_end])
        audit.append(
            {
                "class": class_id,
                "total": n,
                "train": train_end,
                "validation": val_end - train_end,
                "reserved_untouched": n - val_end,
            }
        )

    train_indices = np.concatenate(train_indices)
    val_indices = np.concatenate(val_indices)
    if np.intersect1d(train_indices, val_indices).size:
        raise RuntimeError("Train/validation index overlap.")
    x_train = _take_and_convert(x_mmap, train_indices)
    x_val = _take_and_convert(x_mmap, val_indices)
    return x_train, y[train_indices], x_val, y[val_indices], audit


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, logits = model(x)
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum").cpu())
        correct += int((logits.argmax(1) == y).sum())
        total += int(y.numel())
    return correct / total, loss_sum / total


def augment_adsb(x):
    """Mild long-record augmentation; validation samples never pass here."""
    batch = x.shape[0]
    phase = torch.empty(batch, 1, device=x.device).uniform_(
        -float(np.deg2rad(3.0)), float(np.deg2rad(3.0))
    )
    cosine, sine = torch.cos(phase), torch.sin(phase)
    i_data, q_data = x[:, 0].clone(), x[:, 1].clone()
    out = x.clone()
    out[:, 0] = i_data * cosine - q_data * sine
    out[:, 1] = i_data * sine + q_data * cosine
    shifts = torch.randint(-8, 9, (batch,), device=x.device)
    out = torch.stack(
        [torch.roll(out[index], int(shifts[index]), dims=-1) for index in range(batch)]
    )
    signal_power = out.square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    snr_db = torch.empty(batch, 1, 1, device=x.device).uniform_(35.0, 45.0)
    noise_power = signal_power / torch.pow(10.0, snr_db / 10.0)
    return out + torch.randn_like(out) * torch.sqrt(noise_power)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=r"C:\Users\123\Downloads\Dataset\Dataset")
    parser.add_argument("--save_dir", default="./results/adsb_closedset_dev_seed41")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--model", choices=["original", "adsb_long"], default="original")
    parser.add_argument("--use_adsb_augmentation", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train, y_train, x_val, y_val, split_audit = build_train_validation(args.data_root)

    train_set = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_set = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    class_counts = np.bincount(y_train, minlength=90)
    sample_weights = 1.0 / class_counts[y_train]
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)

    model_factory = ADSBLongClosedSet if args.model == "adsb_long" else ClosedSetSEI
    model = model_factory(num_known_classes=90, feat_dim=args.feat_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_factor(epoch_index):
        if epoch_index < args.warmup_epochs:
            return float(epoch_index + 1) / max(args.warmup_epochs, 1)
        progress = (epoch_index - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    best_acc, best_loss, best_epoch, stale = -1.0, float("inf"), 0, 0
    rows = []
    checkpoint = os.path.join(args.save_dir, "best_closedset_adsb90_validation_only.pth")

    print(f"[ADS-B closed-set dev] device={device} train={len(train_set)} val={len(val_set)}")
    print("[Boundary] X_test_30Class.npy is not opened; reserved base 30% is not loaded.")
    for epoch in range(1, args.epochs + 1):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            train_x = augment_adsb(x) if args.use_adsb_augmentation else x
            _, logits = model(train_x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            correct += int((logits.argmax(1) == y).sum())
            total += int(y.numel())
            loss_sum += float(loss.detach().cpu()) * int(y.numel())
        scheduler.step()

        train_acc = correct / total
        val_acc, val_loss = evaluate(model, val_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]
        improved = val_acc > best_acc + 1e-12 or (
            abs(val_acc - best_acc) <= 1e-12 and val_loss < best_loss
        )
        if improved:
            best_acc, best_loss, best_epoch, stale = val_acc, val_loss, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_known_classes": 90,
                    "feat_dim": args.feat_dim,
                    "selection": {
                        "epoch": epoch,
                        "validation_accuracy": val_acc,
                        "validation_loss": val_loss,
                    },
                    "metadata": {
                        "protocol": "stored_order_60_train_10_validation_30_untouched",
                        "recipe": "CE_label_smoothing_balanced_sampler_adamw_warmup_cosine",
                        "model": args.model,
                        "adsb_augmentation": bool(args.use_adsb_augmentation),
                        "seed": args.seed,
                        "test_files_opened": False,
                    },
                },
                checkpoint,
            )
        else:
            stale += 1

        rows.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / total,
                "train_accuracy": train_acc,
                "validation_loss": val_loss,
                "validation_accuracy": val_acc,
                "lr": lr_now,
                "is_best": improved,
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | train_acc={train_acc:.4f} "
            f"| val_acc={val_acc:.4f} | val_loss={val_loss:.4f} | lr={lr_now:.6g}"
        )
        if stale >= args.patience:
            print(f"[Early stop] no validation improvement for {args.patience} epochs.")
            break

    with open(os.path.join(args.save_dir, "training_history.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(args.save_dir, "development_report.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_validation_accuracy": best_acc,
                "best_validation_loss": best_loss,
                "best_epoch": best_epoch,
                "train_samples": len(train_set),
                "validation_samples": len(val_set),
                "reserved_samples_loaded": False,
                "test_files_opened": False,
                "split_audit": split_audit,
                "configuration": vars(args),
            },
            f,
            indent=2,
        )
    print(f"[Done] best_val_acc={best_acc:.4f} epoch={best_epoch} checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
