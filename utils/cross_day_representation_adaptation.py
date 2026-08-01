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

from utils.improved_closedset_training import SupConProjectionHead, augment_iq_batch


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


def _instance_contrastive_loss(
    projected_one: torch.Tensor,
    projected_two: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """计算 SimCLR/NT-Xent 风格的无标签 instance 对比损失。

    同一个 IQ symbol 的两种增强视图互为正样本，batch 中其它样本作为负样本。
    该损失不需要未知类真值，适合在当前 discovery 域上做轻量自监督适配。
    """

    if projected_one.shape[0] <= 1 or projected_two.shape[0] <= 1:
        return projected_one.sum() * 0.0
    if projected_one.shape != projected_two.shape:
        raise ValueError(
            "instance contrastive views must share shape: "
            f"{tuple(projected_one.shape)} vs {tuple(projected_two.shape)}"
        )

    batch_size = int(projected_one.shape[0])
    features = F.normalize(torch.cat([projected_one, projected_two], dim=0), dim=1)
    logits = features @ features.T / float(max(temperature, 1e-6))
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()
    self_mask = torch.eye(batch_size * 2, device=features.device, dtype=torch.bool)
    logits = logits.masked_fill(self_mask, -1e9)
    positive_index = torch.arange(batch_size * 2, device=features.device)
    positive_index = torch.where(
        positive_index < batch_size,
        positive_index + batch_size,
        positive_index - batch_size,
    )
    return F.cross_entropy(logits, positive_index)


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


def adapt_model_lora_ssl(
    model: torch.nn.Module,
    known_x: np.ndarray,
    known_y: np.ndarray,
    discovery_x: np.ndarray,
    *,
    device: str,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-5,
    ce_weight: float = 1.0,
    instance_weight: float = 0.5,
    teacher_weight: float = 0.2,
    temperature: float = 0.2,
    projection_hidden_dim: int = 128,
    projection_dim: int = 64,
    seed: int = 7,
) -> CrossDayAdaptationResult:
    """在 LoRa 当前 discovery 域上做无标签 instance-level 自监督适配。

    设计边界：
    1. 已知类 CE 只使用 Day1 IQ_1-6 训练标签，用来保住旧类身份判别能力；
    2. 当前轮 discovery 只参与增强视图对比和 teacher 特征锚定，不读取未知真值；
    3. held-out evaluation 完全不进入该过程，因此仍符合 strict 协议。
    """

    _set_seed(seed)
    if len(known_x) == 0 or len(discovery_x) == 0:
        raise ValueError("LoRa SSL adaptation requires non-empty known and discovery data.")

    student = copy.deepcopy(model).to(device)
    teacher = copy.deepcopy(model).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    projector = SupConProjectionHead(
        input_dim=int(getattr(student, "feat_dim", projection_hidden_dim)),
        hidden_dim=int(projection_hidden_dim),
        output_dim=int(projection_dim),
    ).to(device)
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
        drop_last=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projector.parameters()),
        lr=float(lr),
        weight_decay=1e-4,
    )
    stats = {"CE": [], "Instance": [], "Teacher": [], "Total": []}

    student.train()
    projector.train()
    for _ in range(max(1, int(epochs))):
        known_iter = cycle(known_loader)
        epoch_values = {key: [] for key in stats}
        for (xd,) in discovery_loader:
            xk, yk = next(known_iter)
            xk = xk.to(device)
            yk = yk.to(device)
            xd = xd.to(device)

            known_feat, known_logits = student(augment_iq_batch(xk))
            disc_view_one = augment_iq_batch(xd)
            disc_view_two = augment_iq_batch(xd)
            disc_feat_one, _ = student(disc_view_one)
            disc_feat_two, _ = student(disc_view_two)
            with torch.no_grad():
                teacher_feat, _ = teacher(xd)

            ce = F.cross_entropy(known_logits, yk)
            instance = _instance_contrastive_loss(
                projector(disc_feat_one),
                projector(disc_feat_two),
                temperature,
            )
            teacher_anchor = (
                F.mse_loss(F.normalize(disc_feat_one, dim=1), F.normalize(teacher_feat, dim=1))
                + F.mse_loss(F.normalize(disc_feat_two, dim=1), F.normalize(teacher_feat, dim=1))
            ) * 0.5
            total = (
                float(ce_weight) * ce
                + float(instance_weight) * instance
                + float(teacher_weight) * teacher_anchor
            )

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(projector.parameters()),
                max_norm=5.0,
            )
            optimizer.step()

            values = {
                "CE": float(ce.detach().cpu()),
                "Instance": float(instance.detach().cpu()),
                "Teacher": float(teacher_anchor.detach().cpu()),
                "Total": float(total.detach().cpu()),
            }
            for key, value in values.items():
                epoch_values[key].append(value)

        for key in stats:
            stats[key].append(float(np.mean(epoch_values[key])) if epoch_values[key] else float("nan"))

    student.eval()
    diagnostics = {
        "LoRa SSL Adapter": "instance_contrastive_teacher_anchor",
        "LoRa SSL Epochs": int(max(1, int(epochs))),
        "LoRa SSL Batch Size": int(batch_size),
        "LoRa SSL LR": float(lr),
        "LoRa SSL CE Weight": float(ce_weight),
        "LoRa SSL Instance Weight": float(instance_weight),
        "LoRa SSL Teacher Weight": float(teacher_weight),
        "LoRa SSL Temperature": float(temperature),
        "LoRa SSL Known Samples": int(len(known_x)),
        "LoRa SSL Discovery Samples": int(len(discovery_x)),
        "LoRa SSL Final Total": float(stats["Total"][-1]),
        "LoRa SSL Final CE": float(stats["CE"][-1]),
        "LoRa SSL Final Instance": float(stats["Instance"][-1]),
        "LoRa SSL Final Teacher": float(stats["Teacher"][-1]),
    }
    return CrossDayAdaptationResult(student, diagnostics)


def _recording_contrastive_loss(
    projected_features: torch.Tensor,
    recording_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """按 LoRa recording_id 计算 supervised contrastive 风格损失。

    ``recording_id`` 是数据加载阶段可观测的 transmission/recording 元数据，
    不是未知设备真值。相同 recording 内的多个 symbol 互为正样本，用来鼓励
    物理同源片段在表征空间内靠近；不同 recording 作为 batch 内负样本，用来
    防止全部 discovery 样本被压到同一个簇。
    """

    if projected_features.ndim != 2:
        raise ValueError("projected_features must be a 2D tensor")
    if recording_ids.ndim != 1 or recording_ids.shape[0] != projected_features.shape[0]:
        raise ValueError("recording_ids must be a vector aligned with projected_features")

    features = F.normalize(projected_features, dim=1)
    logits = features @ features.T / float(max(temperature, 1e-6))
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()
    self_mask = torch.eye(features.shape[0], device=features.device, dtype=torch.bool)
    positive_mask = recording_ids.unsqueeze(0) == recording_ids.unsqueeze(1)
    positive_mask = positive_mask & ~self_mask
    if not torch.any(positive_mask):
        return projected_features.sum() * 0.0

    logits = logits.masked_fill(self_mask, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not torch.any(valid):
        return projected_features.sum() * 0.0
    mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positive_count.clamp_min(1)
    return -mean_log_prob[valid].mean()


def adapt_model_lora_recording_ssl(
    model: torch.nn.Module,
    known_x: np.ndarray,
    known_y: np.ndarray,
    discovery_x: np.ndarray,
    recording_ids: np.ndarray,
    *,
    device: str,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-5,
    ce_weight: float = 1.0,
    recording_weight: float = 0.5,
    instance_weight: float = 0.2,
    teacher_weight: float = 0.2,
    temperature: float = 0.2,
    projection_hidden_dim: int = 128,
    projection_dim: int = 64,
    seed: int = 7,
) -> CrossDayAdaptationResult:
    """LoRa recording-level 自监督适配。

    该适配器比 instance-level SSL 更强：它把同一 recording 的多个 symbol
    当作正样本组，同时继续用 Day1 已知类 CE 保住旧类边界，并用 teacher
    特征锚定限制漂移。整个过程只接收当前 discovery 的 IQ 与可观测
    ``recording_id``，不读取未知类别标签，也不访问 held-out evaluation。
    """

    _set_seed(seed)
    recording_ids = np.asarray(recording_ids)
    if len(known_x) == 0 or len(discovery_x) == 0:
        raise ValueError("LoRa recording SSL requires non-empty known and discovery data.")
    if recording_ids.shape != (len(discovery_x),):
        raise ValueError(
            "recording_ids must align with discovery_x: "
            f"{recording_ids.shape} vs ({len(discovery_x)},)"
        )

    student = copy.deepcopy(model).to(device)
    teacher = copy.deepcopy(model).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    projector = SupConProjectionHead(
        input_dim=int(getattr(student, "feat_dim", projection_hidden_dim)),
        hidden_dim=int(projection_hidden_dim),
        output_dim=int(projection_dim),
    ).to(device)
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
        TensorDataset(
            torch.as_tensor(discovery_x, dtype=torch.float32),
            torch.as_tensor(recording_ids.astype(np.int64, copy=False), dtype=torch.long),
        ),
        batch_size=max(2, int(batch_size)),
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projector.parameters()),
        lr=float(lr),
        weight_decay=1e-4,
    )
    stats = {"CE": [], "Recording": [], "Instance": [], "Teacher": [], "Total": []}

    student.train()
    projector.train()
    for _ in range(max(1, int(epochs))):
        known_iter = cycle(known_loader)
        epoch_values = {key: [] for key in stats}
        for xd, gid in discovery_loader:
            xk, yk = next(known_iter)
            xk = xk.to(device)
            yk = yk.to(device)
            xd = xd.to(device)
            gid = gid.to(device)

            known_feat, known_logits = student(augment_iq_batch(xk))
            disc_view_one = augment_iq_batch(xd)
            disc_view_two = augment_iq_batch(xd)
            disc_feat_one, _ = student(disc_view_one)
            disc_feat_two, _ = student(disc_view_two)
            with torch.no_grad():
                teacher_feat, _ = teacher(xd)

            ce = F.cross_entropy(known_logits, yk)
            projected_one = projector(disc_feat_one)
            projected_two = projector(disc_feat_two)
            doubled_gid = torch.cat([gid, gid], dim=0)
            recording = _recording_contrastive_loss(
                torch.cat([projected_one, projected_two], dim=0),
                doubled_gid,
                temperature,
            )
            instance = _instance_contrastive_loss(projected_one, projected_two, temperature)
            teacher_anchor = (
                F.mse_loss(F.normalize(disc_feat_one, dim=1), F.normalize(teacher_feat, dim=1))
                + F.mse_loss(F.normalize(disc_feat_two, dim=1), F.normalize(teacher_feat, dim=1))
            ) * 0.5
            total = (
                float(ce_weight) * ce
                + float(recording_weight) * recording
                + float(instance_weight) * instance
                + float(teacher_weight) * teacher_anchor
            )

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(projector.parameters()),
                max_norm=5.0,
            )
            optimizer.step()

            values = {
                "CE": float(ce.detach().cpu()),
                "Recording": float(recording.detach().cpu()),
                "Instance": float(instance.detach().cpu()),
                "Teacher": float(teacher_anchor.detach().cpu()),
                "Total": float(total.detach().cpu()),
            }
            for key, value in values.items():
                epoch_values[key].append(value)

        for key in stats:
            stats[key].append(float(np.mean(epoch_values[key])) if epoch_values[key] else float("nan"))

    student.eval()
    diagnostics = {
        "LoRa Recording SSL Adapter": "recording_contrastive_teacher_anchor",
        "LoRa Recording SSL Epochs": int(max(1, int(epochs))),
        "LoRa Recording SSL Batch Size": int(batch_size),
        "LoRa Recording SSL LR": float(lr),
        "LoRa Recording SSL CE Weight": float(ce_weight),
        "LoRa Recording SSL Recording Weight": float(recording_weight),
        "LoRa Recording SSL Instance Weight": float(instance_weight),
        "LoRa Recording SSL Teacher Weight": float(teacher_weight),
        "LoRa Recording SSL Temperature": float(temperature),
        "LoRa Recording SSL Known Samples": int(len(known_x)),
        "LoRa Recording SSL Discovery Samples": int(len(discovery_x)),
        "LoRa Recording SSL Unique Recordings": int(np.unique(recording_ids).size),
        "LoRa Recording SSL Final Total": float(stats["Total"][-1]),
        "LoRa Recording SSL Final CE": float(stats["CE"][-1]),
        "LoRa Recording SSL Final Recording": float(stats["Recording"][-1]),
        "LoRa Recording SSL Final Instance": float(stats["Instance"][-1]),
        "LoRa Recording SSL Final Teacher": float(stats["Teacher"][-1]),
    }
    return CrossDayAdaptationResult(student, diagnostics)
