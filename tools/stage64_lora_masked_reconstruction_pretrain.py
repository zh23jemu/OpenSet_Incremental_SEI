#!/usr/bin/env python3
"""Stage64：LoRa raw I/Q masked reconstruction 初始化 checkpoint。

该工具用于验证一个更大的 LoRa 表征方向：先用原始 I/Q 的遮挡重建任务
预训练 LoRa Chirp backbone，再只用 Day1 IQ_1-6 已知类训练样本做 CE+SupCon
微调，并用 IQ_7 选择 checkpoint。预训练可以额外使用协议允许的 Day2-4
discovery/enrollment 未标注样本，但不会读取未知类别真值，也不会访问任何
IQ_8-10 held-out evaluation。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.lora25_strict_loader import load_lora25_diffdays_3round
from models.lora_chirp_model import LoRaChirpClosedSet
from utils.improved_closedset_training import (
    RFAugmentConfig,
    SupConProjectionHead,
    TRAINING_RECIPE_VERSION,
    augment_iq_batch,
    evaluate_classifier,
    supervised_contrastive_loss,
)


def _set_seed(seed: int) -> None:
    """固定本工具涉及的随机源，保证 seed7 短验证可复现。"""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _allowed_discovery_unlabeled(splits: dict[str, object]) -> np.ndarray:
    """拼接协议允许的 Day2-4 discovery/enrollment 未标注样本。

    这里只返回 ``X``，不返回 ``y``，也不读取 ``eval_after_r*``。这些样本在
    Stage64 中只用于 masked reconstruction，不能提供任何类别监督。
    """
    keys = ("day2_unknown_round1", "day3_unknown_round2", "day4_unknown_round3")
    arrays = [np.asarray(splits[key]["X"], dtype=np.float32) for key in keys]
    return np.concatenate(arrays, axis=0)


def _checkpoint_metadata(args: argparse.Namespace) -> dict[str, object]:
    """生成 strict 主入口可接受的 checkpoint metadata。"""
    return {
        "dataset_path": os.path.abspath(args.dataset_path),
        "selected_rx_list": [2],
        "known_tx": list(range(10)),
        "training_day_index": 0,
        "development_ratio": 0.70,
        "dataset_profile": "lora25",
        "sample_split_protocol": "lora25_transmission_disjoint_60_10_30_v1",
        "closedset_backbone": "lora_chirp",
        "seed": int(args.seed),
        "use_supcon": True,
        "supcon_weight": float(args.supcon_weight),
        "supcon_temperature": float(args.supcon_temperature),
        "training_recipe_version": TRAINING_RECIPE_VERSION,
        "closedset_validation_fraction": 1.0 / 7.0,
        "rf_augmentation": True,
        "supcon_projection_dim": int(args.projection_dim),
        "supcon_projection_hidden_dim": int(args.projection_hidden_dim),
        "masked_reconstruction_pretrain": {
            "schema_version": "stage64_lora_masked_reconstruction_v1",
            "mask_ratio": float(args.mask_ratio),
            "pretrain_epochs": int(args.pretrain_epochs),
            "finetune_epochs": int(args.finetune_epochs),
            "include_discovery_unlabeled": bool(args.include_discovery_unlabeled),
            "reconstruction_loss": "masked_positions_mse",
        },
    }


class _IQLatentDecoder(nn.Module):
    """把 backbone embedding 解码回原始 I/Q 序列，用于遮挡重建预训练。"""

    def __init__(self, feat_dim: int, channels: int, length: int):
        super().__init__()
        self.channels = int(channels)
        self.length = int(length)
        self.net = nn.Sequential(
            nn.Linear(int(feat_dim), 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, self.channels * self.length),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        decoded = self.net(features)
        return decoded.view(features.shape[0], self.channels, self.length)


def _mask_iq_batch(x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """随机遮挡 I/Q 采样点，返回遮挡后的输入和被遮挡位置 mask。"""
    ratio = float(np.clip(mask_ratio, 0.0, 0.95))
    sample_mask = torch.rand(x.shape[0], 1, x.shape[2], device=x.device) < ratio
    masked = x.masked_fill(sample_mask.expand_as(x), 0.0)
    return masked, sample_mask.expand_as(x)


def train_checkpoint(args: argparse.Namespace) -> dict[str, object]:
    """训练并保存 masked reconstruction + Day1 supervised fine-tune checkpoint。"""
    _set_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    splits = load_lora25_diffdays_3round(args.dataset_path)
    train_x = np.asarray(splits["day1_backbone_train"]["X"], dtype=np.float32)
    train_y = np.asarray(splits["day1_backbone_train"]["y"], dtype=np.int64)
    val_x = np.asarray(splits["day1_known_validation"]["X"], dtype=np.float32)
    val_y = np.asarray(splits["day1_known_validation"]["y"], dtype=np.int64)
    discovery_x = _allowed_discovery_unlabeled(splits) if args.include_discovery_unlabeled else None

    channels = int(train_x.shape[1])
    length = int(train_x.shape[2])
    pretrain_x = train_x if discovery_x is None else np.concatenate([train_x, discovery_x], axis=0)

    model = LoRaChirpClosedSet(num_known_classes=10, feat_dim=args.feat_dim).to(device)
    decoder = _IQLatentDecoder(args.feat_dim, channels, length).to(device)
    projector = SupConProjectionHead(
        input_dim=args.feat_dim,
        hidden_dim=args.projection_hidden_dim,
        output_dim=args.projection_dim,
    ).to(device)
    augmentation_config = RFAugmentConfig()

    pretrain_loader = DataLoader(
        TensorDataset(torch.as_tensor(pretrain_x)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(train_x), torch.as_tensor(train_y, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.as_tensor(val_x), torch.as_tensor(val_y, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    pre_optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(decoder.parameters()),
        lr=args.pretrain_lr,
        weight_decay=1e-4,
    )
    pretrain_history = []
    for epoch in range(1, int(args.pretrain_epochs) + 1):
        model.train()
        decoder.train()
        losses = []
        for (xb,) in pretrain_loader:
            xb = xb.to(device)
            augmented = augment_iq_batch(xb, augmentation_config)
            masked, mask = _mask_iq_batch(augmented, args.mask_ratio)
            feat, _ = model(masked)
            reconstruction = decoder(feat)
            # 只在被遮挡位置计算重建误差，避免模型复制未遮挡输入即可过关。
            loss = ((reconstruction - augmented) ** 2 * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
            pre_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(decoder.parameters()), max_norm=5.0)
            pre_optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": int(epoch), "reconstruction_loss": float(np.mean(losses))}
        pretrain_history.append(row)
        print(f"Pretrain {epoch:03d}/{args.pretrain_epochs:03d} | recon={row['reconstruction_loss']:.6f}")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projector.parameters()),
        lr=args.finetune_lr,
        weight_decay=1e-4,
    )
    best = {"validation_accuracy": -1.0, "validation_loss": float("inf"), "epoch": 0}
    finetune_history = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(args.finetune_epochs) + 1):
        model.train()
        projector.train()
        epoch_rows = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            view_one = augment_iq_batch(xb, augmentation_config)
            view_two = augment_iq_batch(xb, augmentation_config)
            feat_one, logits_one = model(view_one)
            feat_two, _ = model(view_two)
            ce = F.cross_entropy(logits_one, yb)
            supcon = supervised_contrastive_loss(
                torch.cat([projector(feat_one), projector(feat_two)], dim=0),
                torch.cat([yb, yb], dim=0),
                temperature=args.supcon_temperature,
            )
            feature_consistency = F.mse_loss(F.normalize(feat_one, dim=1), F.normalize(feat_two, dim=1))
            total = ce + float(args.supcon_weight) * supcon + float(args.feature_consistency_weight) * feature_consistency
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(projector.parameters()), max_norm=5.0)
            optimizer.step()
            epoch_rows.append(
                {
                    "ce": float(ce.detach().cpu()),
                    "supcon": float(supcon.detach().cpu()),
                    "feature_consistency": float(feature_consistency.detach().cpu()),
                    "total": float(total.detach().cpu()),
                }
            )

        validation_accuracy, validation_loss = evaluate_classifier(model, val_loader, device)
        row = {
            "epoch": int(epoch),
            "validation_accuracy": float(validation_accuracy),
            "validation_loss": float(validation_loss),
            "train_ce": float(np.mean([item["ce"] for item in epoch_rows])),
            "train_supcon": float(np.mean([item["supcon"] for item in epoch_rows])),
            "train_feature_consistency": float(np.mean([item["feature_consistency"] for item in epoch_rows])),
            "train_total": float(np.mean([item["total"] for item in epoch_rows])),
        }
        finetune_history.append(row)
        print(
            f"Finetune {epoch:03d}/{args.finetune_epochs:03d} | "
            f"val_acc={validation_accuracy:.4f} | val_loss={validation_loss:.4f} | "
            f"ce={row['train_ce']:.4f} | feat_cons={row['train_feature_consistency']:.4f}"
        )
        improved = validation_accuracy > best["validation_accuracy"] + 1e-12
        tied_lower_loss = abs(validation_accuracy - best["validation_accuracy"]) <= 1e-12 and validation_loss < best["validation_loss"]
        if improved or tied_lower_loss:
            best = {
                "validation_accuracy": float(validation_accuracy),
                "validation_loss": float(validation_loss),
                "epoch": int(epoch),
            }
            metadata = _checkpoint_metadata(args)
            metadata.update({"rf_augmentation_config": asdict(augmentation_config)})
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_known_classes": 10,
                    "feat_dim": int(args.feat_dim),
                    "metadata": metadata,
                    "selection": best,
                },
                output_path,
            )

    summary = {
        "schema_version": "stage64_lora_masked_reconstruction_pretrain_summary_v1",
        "checkpoint": str(output_path),
        "include_discovery_unlabeled": bool(args.include_discovery_unlabeled),
        "pretrain_samples": int(len(pretrain_x)),
        "day1_train_samples": int(len(train_x)),
        "day1_validation_samples": int(len(val_x)),
        "best": best,
        "pretrain_history": pretrain_history,
        "finetune_history": finetune_history,
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="训练 Stage64 LoRa masked reconstruction checkpoint")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--pretrain-lr", type=float, default=8e-4)
    parser.add_argument("--finetune-lr", type=float, default=8e-4)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--supcon-weight", type=float, default=0.1)
    parser.add_argument("--supcon-temperature", type=float, default=0.2)
    parser.add_argument("--mask-ratio", type=float, default=0.35)
    parser.add_argument("--feature-consistency-weight", type=float, default=0.05)
    parser.add_argument("--include-discovery-unlabeled", action="store_true")
    parser.add_argument("--projection-hidden-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    summary = train_checkpoint(args)
    print(json.dumps({"checkpoint": summary["checkpoint"], "best": summary["best"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
