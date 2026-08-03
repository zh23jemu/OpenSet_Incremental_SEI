"""Stage 42：LoRa 跨天 discovery 全局实例对比预训练。

预训练只读取 Day1 IQ_1-6 和 Day2-4 IQ_1-7 discovery 样本，不读取任何
IQ_8-10 held-out evaluation。先用增强后的同一样本做实例对比学习，再用
Day1 已知标签做 CE + SupCon 微调，最后保存可被 strict 主入口读取的
LoRa Chirp checkpoint。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.lora25_strict_loader import load_lora25_diffdays_3round
from models.lora_chirp_model import LoRaChirpClosedSet
from utils.improved_closedset_training import (
    TRAINING_RECIPE_VERSION,
    RFAugmentConfig,
    augment_iq_batch,
    evaluate_classifier,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _instance_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """计算两视图对称 InfoNCE，正样本是同一条 IQ 片段。"""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / float(temperature)
    labels = torch.arange(len(z1), device=z1.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    )


def _supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """本地实现 CE 微调所需的 SupCon，避免导入完整 strict 实验入口。"""
    features = F.normalize(features, dim=1)
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)
    logits = torch.matmul(features, features.T) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye_mask = torch.ones_like(mask) - torch.eye(len(features), device=features.device)
    mask = mask * eye_mask
    exp_logits = torch.exp(logits) * eye_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    positive_count = mask.sum(dim=1)
    valid = positive_count > 0
    if not torch.any(valid):
        return features.sum() * 0.0
    return -((mask * log_prob).sum(dim=1) / (positive_count + 1e-8))[valid].mean()


class _ProjectionHead(nn.Module):
    """实例对比预训练投影头，训练后不写入最终分类 checkpoint。"""

    def __init__(self, feat_dim: int, projection_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _metadata(args: argparse.Namespace) -> dict[str, object]:
    """生成 strict 主入口需要的协议元数据。"""
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
        "supcon_projection_hidden_dim": int(args.feat_dim),
        "global_ssl": {
            "schema_version": "stage42_lora_global_ssl_v1",
            "pretrain_samples": int(args.pretrain_samples),
            "pretrain_epochs": int(args.pretrain_epochs),
            "temperature": float(args.temperature),
        },
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    """执行全局无标签预训练、Day1 微调并保存最佳 checkpoint。"""
    _set_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    splits = load_lora25_diffdays_3round(args.dataset_path)
    train_x = np.asarray(splits["day1_backbone_train"]["X"], dtype=np.float32)
    train_y = np.asarray(splits["day1_backbone_train"]["y"], dtype=np.int64)
    val_x = np.asarray(splits["day1_known_validation"]["X"], dtype=np.float32)
    val_y = np.asarray(splits["day1_known_validation"]["y"], dtype=np.int64)
    discovery_x = np.concatenate(
        [
            train_x,
            np.asarray(splits["day2_unknown_round1"]["X"], dtype=np.float32),
            np.asarray(splits["day3_unknown_round2"]["X"], dtype=np.float32),
            np.asarray(splits["day4_unknown_round3"]["X"], dtype=np.float32),
        ],
        axis=0,
    )
    args.pretrain_samples = len(discovery_x)

    model = LoRaChirpClosedSet(num_known_classes=10, feat_dim=args.feat_dim).to(device)
    projector = _ProjectionHead(args.feat_dim, args.projection_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projector.parameters()),
        lr=args.pretrain_lr,
        weight_decay=1e-4,
    )
    augment_config = RFAugmentConfig()
    ssl_loader = DataLoader(
        TensorDataset(torch.as_tensor(discovery_x)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed + 101),
    )

    history = []
    model.train()
    projector.train()
    for epoch in range(1, args.pretrain_epochs + 1):
        losses = []
        for (xb,) in ssl_loader:
            xb = xb.to(device)
            view_one = augment_iq_batch(xb, augment_config)
            view_two = augment_iq_batch(xb, augment_config)
            z1 = projector(model(view_one)[0])
            z2 = projector(model(view_two)[0])
            loss = _instance_contrastive_loss(z1, z2, args.temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"phase": "global_ssl", "epoch": epoch, "loss": float(np.mean(losses))})

    # 用 Day1 标签做短监督微调，把无标签预训练的几何结构拉回设备识别目标。
    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(train_x), torch.as_tensor(train_y, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed + 202),
    )
    val_loader = DataLoader(
        TensorDataset(torch.as_tensor(val_x), torch.as_tensor(val_y, dtype=torch.long)),
        batch_size=max(args.batch_size, 256),
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=1e-4)
    best = {"validation_accuracy": -1.0, "validation_loss": float("inf"), "epoch": 0}
    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            view_one = augment_iq_batch(xb, augment_config)
            view_two = augment_iq_batch(xb, augment_config)
            feat_one, logits = model(view_one)
            feat_two, _ = model(view_two)
            loss = F.cross_entropy(logits, yb) + args.supcon_weight * _supervised_contrastive_loss(
                torch.cat([feat_one, feat_two], dim=0),
                torch.cat([yb, yb], dim=0),
                temperature=args.supcon_temperature,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_acc, val_loss = evaluate_classifier(model, val_loader, device)
        history.append({
            "phase": "supervised_finetune",
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "validation_accuracy": float(val_acc),
            "validation_loss": float(val_loss),
        })
        if val_acc > best["validation_accuracy"] or (
            abs(val_acc - best["validation_accuracy"]) < 1e-12 and val_loss < best["validation_loss"]
        ):
            best = {"validation_accuracy": float(val_acc), "validation_loss": float(val_loss), "epoch": epoch}
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_known_classes": 10,
                    "feat_dim": int(args.feat_dim),
                    "metadata": _metadata(args),
                    "selection": best,
                },
                output,
            )

    summary = {
        "schema_version": "stage42_lora_global_ssl_summary_v1",
        "checkpoint": str(args.output),
        "pretrain_samples": int(len(discovery_x)),
        "best": best,
        "history": history,
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="训练 Stage 42 LoRa 全局 discovery 实例对比 checkpoint")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pretrain-epochs", type=int, default=12)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--supcon-weight", type=float, default=0.1)
    parser.add_argument("--supcon-temperature", type=float, default=0.2)
    args = parser.parse_args()
    args.pretrain_samples = 0
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
