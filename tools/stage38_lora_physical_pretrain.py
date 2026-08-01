"""Stage 38：训练 LoRa 物理域自监督初始化 checkpoint。

该工具只使用 LoRa strict 协议允许的 Day1 IQ_1-6 已知类训练样本和
IQ_7 验证样本，不读取 Day1 IQ_8-10 或后续天的 held-out evaluation。

与 Stage 36/37 的“训练期短适配”不同，本阶段直接改变初始 backbone
训练目标：在 CE + SupCon 之外加入 LoRa IQ 物理描述符回归，让模型在
闭集训练时同时学习相位步进、幅度形状和频谱分布。输出 checkpoint 的
metadata 与 strict 主入口兼容，可直接被 `exp_lora25_mvacc_cil_strict.py`
加载并进入后续 GPCC/RADCIL 流程。
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

# 该脚本通常以 `python tools/xxx.py` 的形式从项目根目录运行。此时 Python
# 默认只把 `tools/` 放进 sys.path，远端 Slurm 会找不到 datasets/models/utils。
# 显式加入项目根目录，保证本地、worktree 和 Slurm 环境的导入行为一致。
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


def _physical_descriptors(x: torch.Tensor) -> torch.Tensor:
    """从 IQ 序列中提取可微/可批量计算的 LoRa 物理描述符。

    返回 6 维描述符：
    1. 相邻采样点复数相位差均值；
    2. 相邻采样点复数相位差标准差；
    3. 包络幅度均值；
    4. 包络幅度标准差；
    5. 频谱能量质心；
    6. 频谱带宽。

    这些目标只来自样本自身，不含设备真值；用于约束 backbone 保留 LoRa
    chirp 的物理形状，而不是只拟合闭集分类头。
    """

    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"Expected IQ tensor [B, 2, L], got {tuple(x.shape)}")
    i_data = x[:, 0, :]
    q_data = x[:, 1, :]
    amplitude = torch.sqrt(i_data.square() + q_data.square() + 1e-12)

    real_step = i_data[:, 1:] * i_data[:, :-1] + q_data[:, 1:] * q_data[:, :-1]
    imag_step = q_data[:, 1:] * i_data[:, :-1] - i_data[:, 1:] * q_data[:, :-1]
    phase_step = torch.atan2(imag_step, real_step)

    complex_signal = torch.complex(i_data, q_data)
    spectrum_power = torch.fft.fft(complex_signal, dim=-1).abs().square()
    freq = torch.linspace(0.0, 1.0, spectrum_power.shape[-1], device=x.device, dtype=x.dtype)
    power_sum = spectrum_power.sum(dim=1).clamp_min(1e-12)
    centroid = (spectrum_power * freq.unsqueeze(0)).sum(dim=1) / power_sum
    bandwidth = torch.sqrt(
        (spectrum_power * (freq.unsqueeze(0) - centroid.unsqueeze(1)).square()).sum(dim=1)
        / power_sum
        + 1e-12
    )

    return torch.stack(
        [
            phase_step.mean(dim=1),
            phase_step.std(dim=1, unbiased=False),
            amplitude.mean(dim=1),
            amplitude.std(dim=1, unbiased=False),
            centroid,
            bandwidth,
        ],
        dim=1,
    )


@torch.no_grad()
def _descriptor_stats(x: np.ndarray, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """用 Day1 IQ_1-6 训练样本估计描述符标准化参数。"""

    loader = DataLoader(
        TensorDataset(torch.as_tensor(x, dtype=torch.float32)),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    values = []
    for (xb,) in loader:
        values.append(_physical_descriptors(xb.to(device)).cpu())
    descriptors = torch.cat(values, dim=0)
    mean = descriptors.mean(dim=0)
    std = descriptors.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean.to(device), std.to(device)


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
        "physical_pretrain": {
            "schema_version": "stage38_lora_physical_pretrain_v1",
            "physical_weight": float(args.physical_weight),
            "descriptor_names": [
                "phase_step_mean",
                "phase_step_std",
                "amplitude_mean",
                "amplitude_std",
                "spectrum_centroid",
                "spectrum_bandwidth",
            ],
        },
    }


def train_physical_checkpoint(args: argparse.Namespace) -> dict[str, object]:
    """训练并保存物理域自监督 checkpoint。"""

    _set_seed(args.seed)
    # `auto` 在 GPU 节点上使用 CUDA，本地或 CPU 回落环境使用 CPU，避免把
    # 字符串 "auto" 直接传给 PyTorch 导致设备解析失败。
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    splits = load_lora25_diffdays_3round(args.dataset_path)
    train_x = np.asarray(splits["day1_backbone_train"]["X"], dtype=np.float32)
    train_y = np.asarray(splits["day1_backbone_train"]["y"], dtype=np.int64)
    val_x = np.asarray(splits["day1_known_validation"]["X"], dtype=np.float32)
    val_y = np.asarray(splits["day1_known_validation"]["y"], dtype=np.int64)

    model = LoRaChirpClosedSet(num_known_classes=10, feat_dim=args.feat_dim).to(device)
    projector = SupConProjectionHead(
        input_dim=args.feat_dim,
        hidden_dim=args.projection_hidden_dim,
        output_dim=args.projection_dim,
    ).to(device)
    physical_head = nn.Sequential(
        nn.Linear(args.feat_dim, args.feat_dim),
        nn.SiLU(),
        nn.Linear(args.feat_dim, 6),
    ).to(device)

    descriptor_mean, descriptor_std = _descriptor_stats(train_x, args.batch_size, device)
    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(train_x), torch.as_tensor(train_y, dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(int(args.seed)),
    )
    val_loader = DataLoader(
        TensorDataset(torch.as_tensor(val_x), torch.as_tensor(val_y, dtype=torch.long)),
        batch_size=max(args.batch_size, 256),
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projector.parameters()) + list(physical_head.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    augmentation_config = RFAugmentConfig()
    best = {"validation_accuracy": -1.0, "validation_loss": float("inf"), "epoch": 0}
    history = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        projector.train()
        physical_head.train()
        epoch_rows = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            view_one = augment_iq_batch(xb, augmentation_config)
            view_two = augment_iq_batch(xb, augmentation_config)

            feat_one, logits_one = model(view_one)
            feat_two, _ = model(view_two)
            raw_feat, _ = model(xb)
            descriptor_target = (_physical_descriptors(xb) - descriptor_mean) / descriptor_std
            descriptor_pred = physical_head(raw_feat)

            ce = F.cross_entropy(logits_one, yb)
            supcon = supervised_contrastive_loss(
                torch.cat([projector(feat_one), projector(feat_two)], dim=0),
                torch.cat([yb, yb], dim=0),
                temperature=args.supcon_temperature,
            )
            physical = F.mse_loss(descriptor_pred, descriptor_target)
            total = ce + float(args.supcon_weight) * supcon + float(args.physical_weight) * physical

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(projector.parameters()) + list(physical_head.parameters()),
                max_norm=5.0,
            )
            optimizer.step()
            epoch_rows.append(
                {
                    "ce": float(ce.detach().cpu()),
                    "supcon": float(supcon.detach().cpu()),
                    "physical": float(physical.detach().cpu()),
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
            "train_physical": float(np.mean([item["physical"] for item in epoch_rows])),
            "train_total": float(np.mean([item["total"] for item in epoch_rows])),
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"val_acc={validation_accuracy:.4f} | val_loss={validation_loss:.4f} | "
            f"physical={row['train_physical']:.4f}"
        )
        improved = validation_accuracy > best["validation_accuracy"] + 1e-12
        tied_lower_loss = (
            abs(validation_accuracy - best["validation_accuracy"]) <= 1e-12
            and validation_loss < best["validation_loss"]
        )
        if improved or tied_lower_loss:
            best = {
                "validation_accuracy": float(validation_accuracy),
                "validation_loss": float(validation_loss),
                "epoch": int(epoch),
            }
            metadata = _checkpoint_metadata(args)
            metadata.update(
                {
                    "rf_augmentation_config": asdict(augmentation_config),
                    "physical_descriptor_mean": [float(v) for v in descriptor_mean.detach().cpu().tolist()],
                    "physical_descriptor_std": [float(v) for v in descriptor_std.detach().cpu().tolist()],
                }
            )
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
        "schema_version": "stage38_lora_physical_pretrain_summary_v1",
        "checkpoint": str(output_path),
        "best": best,
        "history": history,
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description="训练 Stage 38 LoRa 物理域自监督 checkpoint")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--supcon-weight", type=float, default=0.1)
    parser.add_argument("--supcon-temperature", type=float, default=0.2)
    parser.add_argument("--physical-weight", type=float, default=0.10)
    parser.add_argument("--projection-hidden-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    summary = train_physical_checkpoint(args)
    print(json.dumps({"checkpoint": summary["checkpoint"], "best": summary["best"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
