"""训练期跨天表征适配。

该模块把跨天一致性从增量后端前移到 discovery 之前。每一轮只使用：

1. Day1 已知类训练样本及其标签；
2. 当前轮 discovery 样本，但不使用其真实标签；
3. Teacher 为原始 closed-set checkpoint，Student 为待适配模型。

目标是保持已知类身份判别能力，同时让当前 discovery 域的两种增强视图
产生稳定表征，并通过 CORAL 风格协方差约束降低跨天统计漂移。该模块
不读取 held-out evaluation 数据，也不生成当前轮伪标签。
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from itertools import cycle
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.improved_closedset_training import augment_iq_batch


@dataclass(frozen=True)
class CrossDayAdaptationResult:
    """返回适配后的模型和无标签训练诊断。"""

    model: torch.nn.Module
    diagnostics: dict[str, Any]


def _set_seed(seed: int) -> None:
    """固定适配阶段的随机源，便于 Slurm seed7 结果复现。"""

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算无标签二阶统计对齐损失。"""

    if source.shape[0] <= 1 or target.shape[0] <= 1:
        return source.sum() * 0.0
    source = source - source.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    source_cov = source.T @ source / float(max(source.shape[0] - 1, 1))
    target_cov = target.T @ target / float(max(target.shape[0] - 1, 1))
    return F.mse_loss(source_cov, target_cov)


def adapt_model_cross_day(
    model: torch.nn.Module,
    known_x: np.ndarray,
    known_y: np.ndarray,
    discovery_x: np.ndarray,
    *,
    device: str,
    epochs: int = 2,
    batch_size: int = 128,
    lr: float = 1e-5,
    consistency_weight: float = 0.5,
    coral_weight: float = 0.05,
    seed: int = 7,
) -> CrossDayAdaptationResult:
    """在当前 discovery 前做训练期跨天表征适配。

    ``known_x`` 和 ``known_y`` 负责防止模型遗忘已知类；``discovery_x``
    只作为无标签目标域，真实未知类别标签不会进入损失。
    """

    _set_seed(seed)
    if len(known_x) == 0 or len(discovery_x) == 0:
        raise ValueError("Cross-day adaptation requires non-empty known and discovery data.")

    student = copy.deepcopy(model).to(device)
    teacher = copy.deepcopy(model).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    known_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(known_x, dtype=torch.float32),
            torch.as_tensor(known_y, dtype=torch.long),
        ),
        batch_size=max(2, int(batch_size)),
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    discovery_loader = DataLoader(
        TensorDataset(torch.as_tensor(discovery_x, dtype=torch.float32)),
        batch_size=max(2, int(batch_size)),
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(lr), weight_decay=1e-4)
    stats = {"CE": [], "Consistency": [], "CORAL": [], "Total": []}

    student.train()
    for _ in range(max(1, int(epochs))):
        known_iter = cycle(known_loader)
        epoch_values = {key: [] for key in stats}
        for (xd,) in discovery_loader:
            xk, yk = next(known_iter)
            xk = xk.to(device)
            yk = yk.to(device)
            xd = xd.to(device)

            known_view = augment_iq_batch(xk)
            discovery_view_one = augment_iq_batch(xd)
            discovery_view_two = augment_iq_batch(xd)

            known_feat, known_logits = student(known_view)
            disc_feat_one, _ = student(discovery_view_one)
            disc_feat_two, _ = student(discovery_view_two)
            with torch.no_grad():
                teacher_disc_feat, _ = teacher(xd)

            ce = F.cross_entropy(known_logits, yk)
            consistency = (
                F.mse_loss(F.normalize(disc_feat_one, dim=1), F.normalize(disc_feat_two, dim=1))
                + F.mse_loss(F.normalize(disc_feat_one, dim=1), F.normalize(teacher_disc_feat, dim=1))
            ) * 0.5
            coral = _coral_loss(known_feat, disc_feat_one)
            total = (
                ce
                + float(consistency_weight) * consistency
                + float(coral_weight) * coral
            )

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()

            values = {
                "CE": float(ce.detach().cpu()),
                "Consistency": float(consistency.detach().cpu()),
                "CORAL": float(coral.detach().cpu()),
                "Total": float(total.detach().cpu()),
            }
            for key, value in values.items():
                epoch_values[key].append(value)

        for key in stats:
            stats[key].append(float(np.mean(epoch_values[key])))

    student.eval()
    diagnostics = {
        "Cross-Day Adapter": "train_consistency_coral",
        "Cross-Day Epochs": int(max(1, int(epochs))),
        "Cross-Day Batch Size": int(batch_size),
        "Cross-Day LR": float(lr),
        "Cross-Day Consistency Weight": float(consistency_weight),
        "Cross-Day CORAL Weight": float(coral_weight),
        "Cross-Day Known Samples": int(len(known_x)),
        "Cross-Day Discovery Samples": int(len(discovery_x)),
        "Cross-Day Final Total": float(stats["Total"][-1]),
        "Cross-Day Final CE": float(stats["CE"][-1]),
        "Cross-Day Final Consistency": float(stats["Consistency"][-1]),
        "Cross-Day Final CORAL": float(stats["CORAL"][-1]),
    }
    return CrossDayAdaptationResult(student, diagnostics)
