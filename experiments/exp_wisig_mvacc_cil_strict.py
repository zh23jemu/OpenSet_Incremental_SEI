"""
Clean WiSig cross-day open-set incremental SEI experiment.

10-known / 3-round protocol:
    Day 1: Tx 0-9 known training
    Day 2: Tx 10-19 unknown discovery round 1
    Day 3: Tx 20-29 unknown discovery round 2
    Day 4: Tx 30-39 unknown discovery round 3

Evaluation:
    Each day is split into 70% discovery/enrollment and 30% evaluation samples.
    Day 1: Tx 0-9 70% -> closed-set training; Tx 0-9 30% -> initial evaluation.
    Day 2: Tx 10-19 70% -> R1 discovery; Tx 0-19 30% -> After R1 evaluation.
    Day 3: Tx 20-29 70% -> R2 discovery; Tx 0-29 30% -> After R2 evaluation.
    Day 4: Tx 30-39 70% -> R3 discovery; Tx 0-39 30% -> After R3 evaluation.

Outputs:
    1) clustering_results.csv
    2) incremental_results.csv
    3) per_round_summary_results.csv
    4) shared_discovery_comparison_results.csv
    5) end_to_end_comparison_results.csv
    6) end_to_end_system_comparison.csv

Methods:
    1. Deep only, no reliability
    2. RF only, no reliability
    3. Graph fusion (raw), no consolidation
    4. No-drop consolidation (ablation)
    5. MV-ACC (final method)

Notes:
    This is the clean final experiment script with only the five core methods.
"""


import os
import sys
import argparse
import json
import random
import copy
import pickle
import itertools
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.manifold import SpectralEmbedding, TSNE

try:
    import umap.umap_ as umap
except ImportError:
    umap = None
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.optimize import linear_sum_assignment

try:
    import hdbscan
except ImportError as e:
    raise ImportError("Please install hdbscan first: pip install hdbscan") from e

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from features.rf_features import extract_rf_features_batch
from utils.classic_feature_gating import select_classic_feature_view, save_selection, local_cross_view_consistency
from utils.discovery_feature_adaptation import adapt_discovery_features
from utils.graph_prototype_discovery_adapter import run_gpcc
from utils.recording_consensus_discovery_adapter import (
    run_recording_consensus_gpcc,
    run_recording_gpcc,
    run_transmission_prototype_gpcc,
)
from utils.cross_day_representation_adaptation import (
    adapt_model_cross_day,
    adapt_model_lora_old_day_supervised,
    adapt_model_lora_recording_ssl,
    adapt_model_lora_ssl,
    load_lora_old_day_calibration,
)
from utils.incremental_visualization import save_seen_class_visualizations, save_paper_method_visualizations
from utils.incremental_metric_learning import cosine_proxy_metric_loss
from models.vup_model import ClosedSetSEI
from models.lora_chirp_model import LoRaChirpClosedSet, LoRaHybridClosedSet
from datasets.lora25_strict_loader import load_lora25_diffdays_3round
from utils.improved_closedset_training import (
    TRAINING_RECIPE_VERSION,
    augment_iq_batch,
    stratified_train_validation_split,
    train_closedset_with_validation,
)


# ============================================================
# Basic utilities
# ============================================================

def set_seed(seed: int = 7):
    """固定 Python/NumPy/PyTorch 随机源，并尽量减少 GPU 端同 seed 波动。"""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # LoRa 原始 I/Q 多窗实验对 seed31 较敏感；这里补齐 cuDNN 确定性设置，
    # 避免同一 seed 在不同 Slurm job 中因卷积算法选择出现可见波动。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        # warn_only 保证遇到少数非确定性算子时给出警告而不是直接杀掉长任务。
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_csv(rows, path):
    ensure_dir(os.path.dirname(path))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def to_dataset(X, y):
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)
    return TensorDataset(X, y)


def load_wisig_crossday_10known_3round(
    dataset_path,
    transpose_to_model=True,
    train_ratio=0.70,
    eq_index=0,
    selected_rx_list=None,
    seed=7,
):
    """
    Load the custom WiSig cross-day dataset and create a leakage-free
    10-known + 3-round 7:3 split.

    Expected dataset structure:
        data[tx_i][rx_i][day_i][eq_i] -> (N, 256, 2)

    Protocol:
        Day 1: Tx 0-9    70% -> known training;      30% -> initial eval on Tx 0-9
        Day 2: Tx 10-19  70% -> R1 discovery;       30% -> After R1 eval on Tx 0-19
        Day 3: Tx 20-29  70% -> R2 discovery;       30% -> After R2 eval on Tx 0-29
        Day 4: Tx 30-39  70% -> R3 discovery;       30% -> After R3 eval on Tx 0-39

    Labels are kept as Tx indices in the custom 40-Tx subset:
        Tx0-9 -> labels 0-9, Tx10-19 -> labels 10-19, etc.
    """
    with open(dataset_path, "rb") as f:
        obj = pickle.load(f)

    data = obj["data"]
    tx_list = obj.get("tx_list", list(range(len(data))))
    rx_list = obj.get("rx_list", list(range(len(data[0]))))
    day_list = obj.get("capture_date_list", ["Day1", "Day2", "Day3", "Day4"])

    n_tx = len(tx_list)
    n_rx = len(rx_list)
    n_day = len(day_list)
    if n_tx < 40 or n_day < 4:
        raise ValueError(f"Expected at least 40 Tx and 4 days, got n_tx={n_tx}, n_day={n_day}")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")

    # Receiver selection:
    #   selected_rx_list=[0]     -> fixed single-receiver protocol
    #   selected_rx_list=[0,1,2] -> original multi-receiver mixed protocol
    if selected_rx_list is None:
        selected_rx_list = list(range(n_rx))
    else:
        selected_rx_list = [int(x) for x in selected_rx_list]
        for rx_i in selected_rx_list:
            if rx_i < 0 or rx_i >= n_rx:
                raise ValueError(f"Invalid rx index {rx_i}; available rx range is 0-{n_rx - 1}")

    def collect(tx_indices, day_i, part):
        """Order-block 70/30 proxy split; does not shuffle adjacent stored samples."""
        xs, ys = [], []
        for tx_i in tx_indices:
            for rx_i in selected_rx_list:
                arr = np.asarray(data[tx_i][rx_i][day_i][eq_index])  # (N, 256, 2)
                split_idx = int(arr.shape[0] * train_ratio)
                if part == "train":
                    arr = arr[:split_idx]
                elif part == "eval":
                    arr = arr[split_idx:]
                else:
                    raise ValueError("part must be 'train' or 'eval'")

                if arr.size == 0:
                    continue
                if transpose_to_model:
                    arr = np.transpose(arr, (0, 2, 1))  # (N, 256, 2) -> (N, 2, 256)
                xs.append(arr.astype(np.float32))
                ys.append(np.full(arr.shape[0], tx_i, dtype=np.int64))

        if len(xs) == 0:
            raise ValueError(f"No samples collected for tx_indices={list(tx_indices)}, day_i={day_i}, part={part}")
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    known_tx = list(range(0, 10))
    r1_tx = list(range(10, 20))
    r2_tx = list(range(20, 30))
    r3_tx = list(range(30, 40))

    # Day 1: initial known training and held-out initial evaluation.
    X_train, y_train = collect(known_tx, day_i=0, part="train")
    X_eval_initial, y_eval_initial = collect(known_tx, day_i=0, part="eval")

    # Day 2: R1 discovery and After-R1 held-out evaluation.
    X_r1, y_r1 = collect(r1_tx, day_i=1, part="train")
    X_eval_r1, y_eval_r1 = collect(known_tx + r1_tx, day_i=1, part="eval")

    # Day 3: R2 discovery and After-R2 held-out evaluation.
    X_r2, y_r2 = collect(r2_tx, day_i=2, part="train")
    X_eval_r2, y_eval_r2 = collect(known_tx + r1_tx + r2_tx, day_i=2, part="eval")

    # Day 4: R3 discovery and After-R3 held-out evaluation.
    X_r3, y_r3 = collect(r3_tx, day_i=3, part="train")
    X_eval_r3, y_eval_r3 = collect(known_tx + r1_tx + r2_tx + r3_tx, day_i=3, part="eval")

    return {
        "day1_known_train": {"X": X_train, "y": y_train, "day": day_list[0], "tx_range": "0-9", "split": "70%"},
        "day1_initial_eval": {"X": X_eval_initial, "y": y_eval_initial, "day": day_list[0], "tx_range": "0-9", "split": "30%"},
        "day2_unknown_round1": {"X": X_r1, "y": y_r1, "day": day_list[1], "tx_range": "10-19", "split": "70%"},
        "day2_eval_after_r1": {"X": X_eval_r1, "y": y_eval_r1, "day": day_list[1], "tx_range": "0-19", "split": "30%"},
        "day3_unknown_round2": {"X": X_r2, "y": y_r2, "day": day_list[2], "tx_range": "20-29", "split": "70%"},
        "day3_eval_after_r2": {"X": X_eval_r2, "y": y_eval_r2, "day": day_list[2], "tx_range": "0-29", "split": "30%"},
        "day4_unknown_round3": {"X": X_r3, "y": y_r3, "day": day_list[3], "tx_range": "30-39", "split": "70%"},
        "day4_eval_after_r3": {"X": X_eval_r3, "y": y_eval_r3, "day": day_list[3], "tx_range": "0-39", "split": "30%"},
        "train_ratio": train_ratio,
        "selected_rx_list": selected_rx_list,
    }


# ============================================================
# Model training and feature extraction
# ============================================================


def supervised_contrastive_loss(features, labels, temperature=0.2):
    """
    Supervised Contrastive Loss for closed-set backbone training.

    This is only used in the initial known-class training stage. It encourages
    samples from the same known Tx to be closer in the embedding space and
    different known Tx classes to be more separated.
    """
    if features is None or labels is None:
        return torch.tensor(0.0, device=features.device if features is not None else "cpu")

    features = F.normalize(features, dim=1)
    labels = labels.contiguous().view(-1, 1)
    batch_size = features.shape[0]

    if batch_size <= 1:
        return features.sum() * 0.0

    mask = torch.eq(labels, labels.T).float().to(features.device)
    logits = torch.div(torch.matmul(features, features.T), float(temperature))

    # Numerical stability.
    logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()

    # Remove self-comparisons.
    logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=features.device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    pos_count = mask.sum(dim=1)
    valid = pos_count > 0
    if valid.sum() == 0:
        return features.sum() * 0.0

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (pos_count + 1e-8)
    loss = -mean_log_prob_pos[valid].mean()
    return loss

def train_closedset_model(train_set, num_classes, feat_dim, epochs, batch_size, lr, device, save_path, use_supcon=False, supcon_weight=0.1, supcon_temperature=0.2, checkpoint_metadata=None, seed=7, validation_fraction=1.0 / 7.0, use_rf_augmentation=True, projection_hidden_dim=128, projection_dim=64, validation_set=None, model_factory=None):
    ensure_dir(os.path.dirname(save_path))
    # 默认保持历史通用 1D ResNet；LoRa Stage 27 可注入 chirp-friendly
    # backbone。两类模型都返回 (feat, logits)，因此后续 RADCIL 扩头不需要
    # 知道具体 backbone 类型。
    model_factory = model_factory or (lambda: ClosedSetSEI(num_known_classes=num_classes, feat_dim=feat_dim))
    train_closedset_with_validation(
        train_set=train_set,
        model_factory=model_factory,
        num_classes=num_classes, feat_dim=feat_dim, epochs=epochs, batch_size=batch_size,
        lr=lr, device=device, save_path=save_path, seed=seed, use_supcon=use_supcon,
        supcon_weight=supcon_weight, supcon_temperature=supcon_temperature,
        checkpoint_metadata=checkpoint_metadata, validation_fraction=validation_fraction,
        use_rf_augmentation=use_rf_augmentation, projection_hidden_dim=projection_hidden_dim,
        projection_dim=projection_dim, validation_set=validation_set,
    )
    return load_closedset_model(save_path, num_classes, feat_dim, device, expected_metadata=checkpoint_metadata, model_factory=model_factory)


def load_closedset_model(checkpoint_path, num_classes, feat_dim, device, expected_metadata=None, model_factory=None):
    if model_factory is None:
        model = ClosedSetSEI(num_known_classes=num_classes, feat_dim=feat_dim).to(device)
    else:
        model = model_factory().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if expected_metadata is not None:
        stored = ckpt.get("metadata") if isinstance(ckpt, dict) else None
        if not stored:
            raise RuntimeError(
                "Checkpoint has no protocol metadata. Re-run with --train_closedset "
                "to create an Rx2/Day1/SupCon-specific checkpoint."
            )
        mismatches = {}
        for key, value in expected_metadata.items():
            stored_value = stored.get(key)
            # 历史通用 ResNet checkpoint 创建时没有 backbone 字段，但其结构
            # 与当前默认 resnet1d 完全一致，允许继续读取；LoRa chirp 或其它
            # 新结构若缺字段仍然必须重新训练，避免静默错载模型。
            if key == "closedset_backbone" and stored_value is None and value == "resnet1d":
                continue
            if stored_value != value:
                mismatches[key] = {"expected": value, "stored": stored_value}
        if mismatches:
            raise RuntimeError(
                f"Checkpoint protocol mismatch: {mismatches}. "
                "Use a new checkpoint or re-run with --train_closedset."
            )
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[Load] checkpoint={checkpoint_path}")
    return model


@torch.no_grad()
def extract_deep_features(model, X, y, batch_size, device):
    loader = DataLoader(to_dataset(X, y), batch_size=batch_size, shuffle=False, num_workers=0)
    feats, ys = [], []
    model.eval()

    for xb, yb in loader:
        xb = xb.to(device)
        feat, _ = model(xb)
        feats.append(feat.detach().cpu().numpy())
        ys.append(yb.detach().cpu().numpy())

    return np.concatenate(feats, axis=0).astype(np.float32), np.concatenate(ys, axis=0).astype(np.int64)


@torch.no_grad()
def _recalibrate_batchnorm(model, X, batch_size, device, passes=1, reset_running_stats=False):
    """用允许的训练/发现样本刷新 BatchNorm 统计量。

    LoRa 的主要风险来自跨天分布漂移。该函数只读取 Day1 train/replay 和
    当前轮 discovery/enrollment IQ，不读取 IQ_8-10 held-out evaluation。
    它不反向传播、不使用标签，只让 BatchNorm 层的 running mean/var 对
    当前可用训练域重新估计；默认关闭，保持历史实验行为不变。
    """
    if X is None or len(X) == 0:
        return model
    bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bn_layers:
        return model
    if reset_running_stats:
        for layer in bn_layers:
            layer.reset_running_stats()
    was_training = model.training
    loader = DataLoader(
        TensorDataset(torch.as_tensor(np.asarray(X, dtype=np.float32))),
        batch_size=int(batch_size), shuffle=False, drop_last=False, num_workers=0,
    )
    model.train()
    for _ in range(max(1, int(passes))):
        for (xb,) in loader:
            model(xb.to(device))
    model.train(was_training)
    return model


# ============================================================
# Clustering metrics and HDBSCAN
# ============================================================

def purity_score(y_true, labels):
    total, correct = 0, 0
    for c in np.unique(labels):
        if c == -1:
            continue
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        _, counts = np.unique(y_true[idx], return_counts=True)
        correct += int(counts.max())
        total += int(len(idx))
    return float(correct / total) if total > 0 else 0.0


def hungarian_cluster_accuracy(y_true, labels):
    """One-to-one cluster accuracy; noise samples count as incorrect."""
    y_true = np.asarray(y_true, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    valid = labels != -1
    if len(y_true) == 0 or not np.any(valid):
        return 0.0
    true_ids = np.unique(y_true[valid])
    cluster_ids = np.unique(labels[valid])
    contingency = np.zeros((len(true_ids), len(cluster_ids)), dtype=np.int64)
    true_pos = {int(v): i for i, v in enumerate(true_ids)}
    cluster_pos = {int(v): i for i, v in enumerate(cluster_ids)}
    for yt, yc in zip(y_true[valid], labels[valid]):
        contingency[true_pos[int(yt)], cluster_pos[int(yc)]] += 1
    row_ind, col_ind = linear_sum_assignment(-contingency)
    return float(contingency[row_ind, col_ind].sum() / len(y_true))


def clustering_metrics(y_true, labels):
    clusters = [c for c in np.unique(labels) if c != -1]
    return {
        "Clusters": int(len(clusters)),
        "Noise": float(np.mean(labels == -1)),
        "NMI": float(normalized_mutual_info_score(y_true, labels)),
        "ARI": float(adjusted_rand_score(y_true, labels)),
        "Purity": float(purity_score(y_true, labels)),
        "Hungarian Acc": hungarian_cluster_accuracy(y_true, labels),
    }


def run_hdbscan(features, min_cluster_size, min_samples):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(features)
    probs = getattr(clusterer, "probabilities_", None)
    return labels.astype(np.int64), probs


# ============================================================
# Feature spaces and graph fusion
# ============================================================

def clean_scale(features):
    z = StandardScaler().fit_transform(np.asarray(features, dtype=np.float32))
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def topk_graph(S, top_k):
    n = S.shape[0]
    k = min(top_k, n)
    out = np.zeros_like(S, dtype=np.float32)
    for i in range(n):
        idx = np.argsort(S[i])[-k:]
        out[i, idx] = S[i, idx]
    out = np.maximum(out, out.T)
    np.fill_diagonal(out, 1.0)
    return out.astype(np.float32)


def cosine_graph(features, top_k):
    z = normalize(features, norm="l2", axis=1)
    S = cosine_similarity(z)
    S = np.clip((S + 1.0) / 2.0, 0.0, 1.0)
    return topk_graph(S, top_k)


def rbf_graph(features, top_k):
    D = euclidean_distances(features, features)
    nz = D[D > 0]
    sigma = float(np.median(nz)) if len(nz) > 0 else 1.0
    S = np.exp(-(D ** 2) / (2.0 * sigma ** 2 + 1e-8))
    return topk_graph(np.clip(S, 0.0, 1.0), top_k)


def build_bounded_edge_adaptive_fusion(G_deep, G_rf, alpha_min=0.65, alpha_max=0.95, eps=1e-8, rf_reliability=None):
    """
    Bounded edge-wise adaptive graph fusion.

    Each sample-pair edge receives its own deep/RF fusion weight.
    The deep-view weight is constrained to [alpha_min, alpha_max],
    so RF can provide complementary evidence without dominating the graph.

    Formula:
        rf_support_ij = G_rf_ij / (G_deep_ij + G_rf_ij + eps)
        alpha_ij = alpha_max - (alpha_max - alpha_min) * rf_support_ij
        G_fused_ij = alpha_ij * G_deep_ij + (1 - alpha_ij) * G_rf_ij
    """
    if not (0.0 <= alpha_min <= alpha_max <= 1.0):
        raise ValueError("alpha_min and alpha_max must satisfy 0 <= alpha_min <= alpha_max <= 1")

    G_deep = np.asarray(G_deep, dtype=np.float32)
    G_rf = np.asarray(G_rf, dtype=np.float32)

    G_deep = np.clip(G_deep, 0.0, 1.0).astype(np.float32, copy=False)
    G_rf = np.clip(G_rf, 0.0, 1.0).astype(np.float32, copy=False)

    # tmp is first rf_support, then alpha_ij, then RF weight.
    tmp = G_deep + G_rf
    tmp += eps
    np.divide(G_rf, tmp, out=tmp)

    # alpha_ij: deep-view weight for every edge.
    tmp *= -(alpha_max - alpha_min)
    tmp += alpha_max
    np.clip(tmp, alpha_min, alpha_max, out=tmp)

    if rf_reliability is not None:
        reliability = np.clip(np.asarray(rf_reliability, dtype=np.float32), 0.0, 1.0)
        edge_reliability = np.sqrt(np.outer(reliability, reliability))
        # Local CF-LCG: unreliable RF neighborhoods transfer their weight to
        # the deep view edge-by-edge, without rejecting any sample.
        tmp += (1.0 - tmp) * (1.0 - edge_reliability)

    # Useful lightweight diagnostics without saving the full NxN weight matrix.
    edge_mask = ((G_deep > 0.0) | (G_rf > 0.0))
    if edge_mask.shape[0] == edge_mask.shape[1]:
        np.fill_diagonal(edge_mask, False)
    if np.any(edge_mask):
        vals = tmp[edge_mask]
        weight_stats = {
            "Adaptive Weight Mean": float(np.mean(vals)),
            "Adaptive Weight Min": float(np.min(vals)),
            "Adaptive Weight Max": float(np.max(vals)),
        }
    else:
        weight_stats = {
            "Adaptive Weight Mean": float(np.mean(tmp)),
            "Adaptive Weight Min": float(np.min(tmp)),
            "Adaptive Weight Max": float(np.max(tmp)),
        }
    if rf_reliability is not None:
        weight_stats["Local RF Consistency Mean"] = float(np.mean(rf_reliability))
        weight_stats["Local RF Consistency Min"] = float(np.min(rf_reliability))

    # G_fused = alpha_ij * G_deep + (1 - alpha_ij) * G_rf
    G_fused = G_deep.copy()
    G_fused *= tmp
    tmp *= -1.0
    tmp += 1.0
    G_fused += tmp * G_rf

    G_fused = np.maximum(G_fused, G_fused.T)
    np.fill_diagonal(G_fused, 1.0)
    return G_fused.astype(np.float32), weight_stats


def graph_embedding(z_deep, z_rf, alpha, top_k, graph_dim, seed, adaptive_fusion=False, alpha_min=0.65, alpha_max=0.95, rf_reliability=None):
    Gd = cosine_graph(z_deep, top_k)
    Gr = rbf_graph(z_rf, top_k)

    if adaptive_fusion:
        G, fusion_stats = build_bounded_edge_adaptive_fusion(
            Gd, Gr, alpha_min=alpha_min, alpha_max=alpha_max, rf_reliability=rf_reliability
        )
    else:
        G = np.maximum(alpha * Gd + (1.0 - alpha) * Gr, 0.0)
        G = np.maximum(G, G.T)
        np.fill_diagonal(G, 1.0)
        fusion_stats = {
            "Adaptive Weight Mean": np.nan,
            "Adaptive Weight Min": np.nan,
            "Adaptive Weight Max": np.nan,
        }

    n_comp = min(graph_dim, G.shape[0] - 2)
    emb = SpectralEmbedding(n_components=n_comp, affinity="precomputed", random_state=seed).fit_transform(G)
    return clean_scale(emb), fusion_stats


def cflcg_mode_for_method(method):
    if method == "No CF-LCG (MV-ACC)":
        return "none"
    if method == "Global CF-LCG (MV-ACC)":
        return "global"
    if method == "MV-ACC":
        return "local"
    return "none"


def extract_rf_view(X, args, cflcg_mode="none"):
    if cflcg_mode == "global" and not getattr(args, "cflcg_gate_open", True):
        # The global ablation is a true binary gate: a failed Day-1 criterion
        # removes the RF view instead of silently falling back to raw RF.
        return np.zeros((len(X), 24), dtype=np.float32)
    if cflcg_mode in {"global", "local"} and getattr(args, "cflcg_extractor", None) is not None:
        return args.cflcg_extractor(X)
    return extract_rf_features_batch(X)


def build_round_features(X_round, Z_round, args, cflcg_mode="none"):
    rf = extract_rf_view(X_round, args, cflcg_mode)
    adapter_name = str(getattr(args, "discovery_feature_adapter", "none")).lower()
    if adapter_name == "none":
        # 默认路径保持历史 clean_scale 行为，确保已有实验不受新开关影响。
        z_deep = clean_scale(Z_round)
        adapter_diagnostics = {"Discovery Feature Adapter": "none"}
    else:
        adapter_result = adapt_discovery_features(
            getattr(args, "discovery_adapter_known_Z", None),
            getattr(args, "discovery_adapter_known_y", None),
            Z_round,
            method=adapter_name,
            smooth_k=getattr(args, "discovery_adapter_smooth_k", 12),
            smooth_weight=getattr(args, "discovery_adapter_smooth_weight", 0.20),
            repulsion_weight=getattr(args, "discovery_adapter_repulsion_weight", 0.15),
        )
        # discovery 特征适配只作用在当前轮 deep view；RF view 仍按原逻辑抽取，
        # graph view 则由适配后的 deep view 与 RF view 共同构造。
        z_deep = adapter_result.features
        adapter_diagnostics = adapter_result.diagnostics
    z_rf = clean_scale(rf)
    local_consistency = local_cross_view_consistency(z_deep, z_rf, args.cflcg_local_k) if cflcg_mode == "local" else None
    z_graph, fusion_stats = graph_embedding(
        z_deep,
        z_rf,
        args.alpha,
        args.top_k,
        args.graph_dim,
        args.seed,
        adaptive_fusion=args.adaptive_fusion,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        rf_reliability=local_consistency,
    )
    return {
        "deep": z_deep,
        "rf": z_rf,
        "graph": z_graph,
        "hybrid": np.concatenate([z_deep, z_rf], axis=1).astype(np.float32),
        "fusion_stats": fusion_stats,
        "local_rf_consistency": local_consistency,
        "feature_adapter_diagnostics": adapter_diagnostics,
    }


# ============================================================
# Reliability and merge
# ============================================================

def reliability_filter(features, labels, probs, min_cluster_size, min_prob, threshold):
    details = []
    cluster_ids = [c for c in np.unique(labels) if c != -1]
    if not cluster_ids:
        return [], details

    stats = []
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        z = features[idx]
        center = z.mean(axis=0)
        dist = np.linalg.norm(z - center, axis=1)
        stats.append({
            "cluster_id": int(cid),
            "indices": idx,
            "size": int(len(idx)),
            "center": center,
            "radius": float(np.mean(dist)),
            "prob": float(np.mean(probs[idx])) if probs is not None else 1.0,
        })

    centers = np.stack([s["center"] for s in stats], axis=0)
    if len(stats) > 1:
        D = euclidean_distances(centers, centers)
        np.fill_diagonal(D, np.inf)
        for i, s in enumerate(stats):
            s["sep"] = float(np.min(D[i]))
    else:
        stats[0]["sep"] = 1.0

    radius_ref = float(np.median([s["radius"] for s in stats]) + 1e-8)
    sep_ref = float(np.median([s["sep"] for s in stats]) + 1e-8)

    accepted = []
    for s in stats:
        if s["size"] < min_cluster_size or s["prob"] < min_prob:
            score, ok = 0.0, False
        else:
            compact = np.exp(-s["radius"] / radius_ref)
            sep = 1.0 - np.exp(-s["sep"] / sep_ref)
            score = float(np.exp(np.mean(np.log(np.clip([compact, sep, s["prob"]], 1e-8, 1.0)))))
            ok = score >= threshold

        item = {"cluster_id": int(s["cluster_id"]), "size": int(s["size"]), "reliability_score": float(score), "accepted": bool(ok)}
        details.append(item)
        if ok:
            accepted.append(int(s["cluster_id"]))

    return accepted, details


def all_non_noise_as_accepted(labels):
    return [int(c) for c in np.unique(labels) if c != -1], [
        {"cluster_id": int(c), "accepted": True, "reliability_score": 1.0, "size": int(np.sum(labels == c))}
        for c in np.unique(labels) if c != -1
    ]


def merge_accepted_clusters(labels, accepted_ids, graph_feat, rf_feat, threshold, mutual_nearest=True):
    accepted_ids = list(accepted_ids)
    if len(accepted_ids) <= 1:
        return labels.copy(), []

    protos_g, protos_r, valid = [], [], []
    for cid in accepted_ids:
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue
        protos_g.append(graph_feat[idx].mean(axis=0))
        protos_r.append(rf_feat[idx].mean(axis=0))
        valid.append(int(cid))

    if len(valid) <= 1:
        return labels.copy(), []

    Pg = normalize(np.stack(protos_g, axis=0), axis=1)
    Sg = np.clip((Pg @ Pg.T + 1.0) / 2.0, 0.0, 1.0)

    Dr = euclidean_distances(np.stack(protos_r, axis=0), np.stack(protos_r, axis=0))
    nz = Dr[Dr > 0]
    sigma = float(np.median(nz)) if len(nz) > 0 else 1.0
    Sr = np.exp(-(Dr ** 2) / (2.0 * sigma ** 2 + 1e-8))

    S = np.sqrt(np.clip(Sg, 1e-8, 1.0) * np.clip(Sr, 1e-8, 1.0))
    np.fill_diagonal(S, 0.0)
    nearest = np.argmax(S, axis=1)

    parent = {cid: cid for cid in valid}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
            return True
        return False

    merge_rows = []
    for i, j in enumerate(nearest):
        if i == j or S[i, j] < threshold:
            continue
        if mutual_nearest and nearest[j] != i:
            continue
        ci, cj = valid[i], valid[j]
        if union(ci, cj):
            merge_rows.append({"cluster_i": int(ci), "cluster_j": int(cj), "merge_score": float(S[i, j])})

    merged = labels.copy()
    for cid in valid:
        merged[labels == cid] = find(cid)

    non_noise = merged != -1
    unique = sorted(np.unique(merged[non_noise]).tolist())
    remap = {old: new for new, old in enumerate(unique)}
    final = np.full_like(merged, -1)
    for old, new in remap.items():
        final[merged == old] = new

    return final.astype(np.int64), merge_rows




# ============================================================
# No-drop iterative cluster merging
# ============================================================

def _merge_labels_by_pairs_no_drop(labels, pairs):
    """
    Merge cluster IDs according to pairs. This never deletes samples.
    Noise labels (-1) stay as noise. Non-noise samples remain assigned.
    """
    labels = np.asarray(labels, dtype=np.int64)
    valid = [int(c) for c in np.unique(labels) if c != -1]
    parent = {c: c for c in valid}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        if a not in parent or b not in parent:
            return False
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[max(ra, rb)] = min(ra, rb)
        return True

    merge_count = 0
    for a, b in pairs:
        if union(int(a), int(b)):
            merge_count += 1

    merged = labels.copy()
    for c in valid:
        merged[labels == c] = find(c)

    # Remap non-noise clusters to consecutive IDs for clean downstream enrollment.
    final = np.full_like(merged, -1)
    non_noise_ids = sorted(np.unique(merged[merged != -1]).tolist())
    remap = {old: new for new, old in enumerate(non_noise_ids)}
    for old, new in remap.items():
        final[merged == old] = new

    return final.astype(np.int64), merge_count


def iterative_merge_clusters_no_drop(labels, feats, args, max_rounds=2, threshold=None):
    """
    Pure iterative cluster merging without reliability filtering or sample deletion.

    Goal:
        reduce over-clustering while preserving all non-noise samples.

    Difference from Full method / MV-IARC:
        - no reliability filtering
        - no rejected clusters
        - no uncertain-sample deletion
        - no secondary sample-level HDBSCAN
        - only merges existing HDBSCAN clusters based on multi-view prototype similarity

    Similarity:
        deep prototype cosine + RF prototype RBF + graph prototype cosine
    """
    working = np.asarray(labels, dtype=np.int64).copy()
    merge_threshold = float(args.iter_merge_threshold if threshold is None else threshold)
    total_merges = 0

    for merge_round in range(int(max_rounds)):
        cluster_ids = [int(c) for c in np.unique(working) if c != -1]
        if len(cluster_ids) <= 1:
            break

        protos_deep, protos_rf, protos_graph = [], [], []
        valid = []
        for cid in cluster_ids:
            idx = np.where(working == cid)[0]
            if len(idx) == 0:
                continue
            protos_deep.append(feats["deep"][idx].mean(axis=0))
            protos_rf.append(feats["rf"][idx].mean(axis=0))
            protos_graph.append(feats["graph"][idx].mean(axis=0))
            valid.append(cid)

        if len(valid) <= 1:
            break

        Pd = normalize(np.stack(protos_deep, axis=0), axis=1)
        Pg = normalize(np.stack(protos_graph, axis=0), axis=1)
        S_deep = np.clip((Pd @ Pd.T + 1.0) / 2.0, 0.0, 1.0)
        S_graph = np.clip((Pg @ Pg.T + 1.0) / 2.0, 0.0, 1.0)

        Pr = np.stack(protos_rf, axis=0)
        Dr = euclidean_distances(Pr, Pr)
        nz = Dr[Dr > 0]
        sigma = float(np.median(nz)) if len(nz) > 0 else 1.0
        S_rf = np.exp(-(Dr ** 2) / (2.0 * sigma ** 2 + 1e-8))

        w_deep = float(args.iter_merge_lambda_deep)
        w_rf = float(args.iter_merge_lambda_rf)
        w_graph = float(args.iter_merge_lambda_graph)
        w_sum = max(w_deep + w_rf + w_graph, 1e-8)
        S = (w_deep * S_deep + w_rf * S_rf + w_graph * S_graph) / w_sum
        np.fill_diagonal(S, 0.0)

        nearest = np.argmax(S, axis=1)
        pairs = []
        for i, j in enumerate(nearest):
            if i == j:
                continue
            if S[i, j] < merge_threshold:
                continue
            if (not args.no_mutual_nearest) and nearest[j] != i:
                continue
            pairs.append((valid[i], valid[j]))

        if not pairs:
            break

        working, merge_count = _merge_labels_by_pairs_no_drop(working, pairs)
        total_merges += int(merge_count)
        if merge_count == 0:
            break

    stats = {
        "Iter Merge Rounds": int(max_rounds),
        "Iter Merge Total Merges": int(total_merges),
        "Iter Merge Threshold": float(merge_threshold),
    }
    return working.astype(np.int64), stats


def merge_closest_clusters_to_target_no_drop(labels, feats, args, target_count):
    """将过聚类结果按多视图最近原型合并到协议指定的目标簇数。

    该步骤只读取 discovery 样本的深度、RF 和图表征，以及实验协议预先
    声明的每轮新增类别数；不读取 discovery 或 held-out evaluation 真值。
    当当前簇数不高于目标值时保持原结果，避免把欠聚类进一步恶化。
    """
    working = np.asarray(labels, dtype=np.int64).copy()
    target_count = int(target_count)
    if target_count <= 0:
        raise ValueError("target_count must be positive.")

    total_merges = 0
    last_similarity = np.nan
    while len([c for c in np.unique(working) if c != -1]) > target_count:
        cluster_ids = [int(c) for c in np.unique(working) if c != -1]
        protos_deep, protos_rf, protos_graph = [], [], []
        for cid in cluster_ids:
            idx = np.where(working == cid)[0]
            protos_deep.append(feats["deep"][idx].mean(axis=0))
            protos_rf.append(feats["rf"][idx].mean(axis=0))
            protos_graph.append(feats["graph"][idx].mean(axis=0))

        pd = normalize(np.stack(protos_deep, axis=0), axis=1)
        pg = normalize(np.stack(protos_graph, axis=0), axis=1)
        similarity_deep = np.clip((pd @ pd.T + 1.0) / 2.0, 0.0, 1.0)
        similarity_graph = np.clip((pg @ pg.T + 1.0) / 2.0, 0.0, 1.0)
        pr = np.stack(protos_rf, axis=0)
        distances_rf = euclidean_distances(pr, pr)
        positive = distances_rf[distances_rf > 0]
        sigma = float(np.median(positive)) if len(positive) > 0 else 1.0
        similarity_rf = np.exp(-(distances_rf ** 2) / (2.0 * sigma ** 2 + 1e-8))

        weight_deep = float(args.mvacc_lambda_deep)
        weight_rf = float(args.mvacc_lambda_rf)
        weight_graph = float(args.mvacc_lambda_graph)
        weight_sum = max(weight_deep + weight_rf + weight_graph, 1e-8)
        similarity = (
            weight_deep * similarity_deep
            + weight_rf * similarity_rf
            + weight_graph * similarity_graph
        ) / weight_sum
        np.fill_diagonal(similarity, -np.inf)
        left, right = np.unravel_index(np.argmax(similarity), similarity.shape)
        last_similarity = float(similarity[left, right])
        keep_id = min(cluster_ids[left], cluster_ids[right])
        merge_id = max(cluster_ids[left], cluster_ids[right])
        working[working == merge_id] = keep_id
        total_merges += 1

    return working.astype(np.int64), {
        "Target Cluster Count": target_count,
        "Target Count Merges": int(total_merges),
        "Target Count Last Merge Similarity": last_similarity,
    }


def assign_noise_samples_no_drop(labels, feats, args):
    """Assign every HDBSCAN noise sample without using ground-truth labels."""
    working = np.asarray(labels, dtype=np.int64).copy()
    noise_idx = np.where(working == -1)[0]
    stats = {
        "Noise Before Reassignment": int(len(noise_idx)),
        "Noise Reassigned": 0,
        "Assignment Coverage": 1.0 if len(working) == 0 else float(np.mean(working != -1)),
        "Assignment Score Mean": np.nan,
        "Assignment Score Min": np.nan,
        "No-cluster Fallback": False,
    }
    if len(noise_idx) == 0:
        stats["Assignment Coverage"] = 1.0
        return working, stats

    cluster_ids = [int(c) for c in np.unique(working) if c != -1]
    if not cluster_ids:
        working[:] = 0
        stats.update({"Noise Reassigned": int(len(noise_idx)), "Assignment Coverage": 1.0, "No-cluster Fallback": True})
        return working, stats

    protos_deep, protos_rf, protos_graph = [], [], []
    for cid in cluster_ids:
        idx = np.where(working == cid)[0]
        protos_deep.append(feats["deep"][idx].mean(axis=0))
        protos_rf.append(feats["rf"][idx].mean(axis=0))
        protos_graph.append(feats["graph"][idx].mean(axis=0))

    Pd = normalize(np.stack(protos_deep, axis=0), axis=1)
    Pg = normalize(np.stack(protos_graph, axis=0), axis=1)
    Xd = normalize(feats["deep"][noise_idx], axis=1)
    Xg = normalize(feats["graph"][noise_idx], axis=1)
    S_deep = np.clip((Xd @ Pd.T + 1.0) / 2.0, 0.0, 1.0)
    S_graph = np.clip((Xg @ Pg.T + 1.0) / 2.0, 0.0, 1.0)

    Pr = np.stack(protos_rf, axis=0)
    Dr = euclidean_distances(feats["rf"][noise_idx], Pr)
    positive = Dr[Dr > 0]
    sigma = float(np.median(positive)) if len(positive) > 0 else 1.0
    S_rf = np.exp(-(Dr ** 2) / (2.0 * sigma ** 2 + 1e-8))

    w_deep = float(args.nodrop_assign_lambda_deep)
    w_rf = float(args.nodrop_assign_lambda_rf)
    w_graph = float(args.nodrop_assign_lambda_graph)
    w_sum = max(w_deep + w_rf + w_graph, 1e-8)
    scores = (w_deep * S_deep + w_rf * S_rf + w_graph * S_graph) / w_sum
    best_pos = np.argmax(scores, axis=1)
    best_scores = scores[np.arange(len(noise_idx)), best_pos]
    working[noise_idx] = np.asarray(cluster_ids, dtype=np.int64)[best_pos]
    stats.update({
        "Noise Reassigned": int(len(noise_idx)),
        "Assignment Coverage": float(np.mean(working != -1)),
        "Assignment Score Mean": float(np.mean(best_scores)),
        "Assignment Score Min": float(np.min(best_scores)),
    })
    if np.any(working == -1):
        raise RuntimeError("No-drop assignment failed: some samples still have label -1")
    return working.astype(np.int64), stats


def adaptive_split_large_clusters_no_drop(labels, feats, args):
    """Split statistically oversized clusters without using labels or a target K.

    Density clustering can occasionally merge two transmitters when a backbone
    seed changes local geometry.  After no-drop consolidation, clusters much
    larger than the median are tested with a two-way K-means split in a joint
    deep/RF/graph space.  A split is retained only when both parts are large
    enough and their within-cluster silhouette exceeds a fixed threshold.
    """
    working = np.asarray(labels, dtype=np.int64).copy()
    z = np.concatenate([
        0.75 * normalize(feats["deep"], axis=1),
        0.25 * normalize(feats["rf"], axis=1),
        normalize(feats["graph"], axis=1),
    ], axis=1).astype(np.float32)
    next_id = int(working.max()) + 1 if len(working) else 0
    total_splits = 0
    accepted_silhouettes = []

    for _ in range(int(args.mvacc_split_rounds)):
        cluster_ids = sorted(np.unique(working).tolist())
        sizes = np.asarray([np.sum(working == c) for c in cluster_ids], dtype=np.int64)
        if len(sizes) <= 1:
            break
        median_size = float(np.median(sizes))
        changed = False

        for cid, size in sorted(zip(cluster_ids, sizes), key=lambda item: item[1], reverse=True):
            if size <= float(args.mvacc_split_size_factor) * median_size:
                continue
            idx = np.where(working == cid)[0]
            sub = KMeans(n_clusters=2, n_init=10, random_state=args.seed).fit_predict(z[idx])
            counts = np.bincount(sub, minlength=2)
            min_part = max(20, int(round(float(args.mvacc_split_min_part_ratio) * median_size)))
            if int(counts.min()) < min_part:
                continue
            score = float(silhouette_score(z[idx], sub, metric="euclidean"))
            if score < float(args.mvacc_split_silhouette):
                continue

            working[idx[sub == 1]] = next_id
            next_id += 1
            total_splits += 1
            accepted_silhouettes.append(score)
            changed = True

        if not changed:
            break

    final = np.empty_like(working)
    for new_id, old_id in enumerate(sorted(np.unique(working).tolist())):
        final[working == old_id] = new_id
    return final.astype(np.int64), {
        "Adaptive Split Count": int(total_splits),
        "Adaptive Split Size Factor": float(args.mvacc_split_size_factor),
        "Adaptive Split Silhouette Threshold": float(args.mvacc_split_silhouette),
        "Adaptive Split Silhouette Mean": (
            float(np.mean(accepted_silhouettes)) if accepted_silhouettes else np.nan
        ),
    }


# ============================================================
# MV-IARC: Multi-view Intra-round Adaptive Re-Clustering
# ============================================================

def _safe_cosine_proto(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(1, -1)
    b = np.asarray(b, dtype=np.float32).reshape(1, -1)
    sim = float((l2norm(a) @ l2norm(b).T)[0, 0])
    # Map cosine similarity from [-1, 1] to [0, 1] for stable fusion with RBF similarity.
    return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))


def _rf_rbf_similarity(p_s, p_r, sigma):
    d = float(np.linalg.norm(np.asarray(p_s, dtype=np.float32) - np.asarray(p_r, dtype=np.float32)))
    return float(np.exp(-(d ** 2) / (2.0 * sigma ** 2 + 1e-8)))


def _cluster_indices(labels, cid):
    return np.where(np.asarray(labels) == cid)[0]


def _cluster_proto(feat, idx):
    return np.asarray(feat[idx], dtype=np.float32).mean(axis=0)


def _estimate_rf_sigma(rf_feat, label_source, reliable_ids, secondary_indices=None):
    protos = []
    for cid in reliable_ids:
        idx = _cluster_indices(label_source, cid)
        if len(idx) > 0:
            protos.append(_cluster_proto(rf_feat, idx))
    if secondary_indices is not None and len(secondary_indices) > 0:
        protos.append(_cluster_proto(rf_feat, secondary_indices))
    if len(protos) <= 1:
        return 1.0
    D = euclidean_distances(np.stack(protos, axis=0), np.stack(protos, axis=0))
    nz = D[D > 0]
    return float(np.median(nz)) if len(nz) > 0 else 1.0


def multi_view_cluster_similarity(sec_idx, ref_idx, feats, args, rf_sigma):
    """
    Similarity between a secondary cluster and an existing reliable cluster.
    Uses deep prototype cosine, RF prototype RBF similarity and graph-embedding cosine.
    """
    pds = _cluster_proto(feats["deep"], sec_idx)
    pdr = _cluster_proto(feats["deep"], ref_idx)
    prs = _cluster_proto(feats["rf"], sec_idx)
    prr = _cluster_proto(feats["rf"], ref_idx)
    pgs = _cluster_proto(feats["graph"], sec_idx)
    pgr = _cluster_proto(feats["graph"], ref_idx)

    s_deep = _safe_cosine_proto(pds, pdr)
    s_rf = _rf_rbf_similarity(prs, prr, rf_sigma)
    s_graph = _safe_cosine_proto(pgs, pgr)

    w_deep = float(args.iarc_lambda_deep)
    w_rf = float(args.iarc_lambda_rf)
    w_graph = float(args.iarc_lambda_graph)
    w_sum = max(w_deep + w_rf + w_graph, 1e-8)
    score = (w_deep * s_deep + w_rf * s_rf + w_graph * s_graph) / w_sum
    return float(score), {"s_deep": s_deep, "s_rf": s_rf, "s_graph": s_graph}


def intra_round_adaptive_reclustering(labels, accepted_ids, discovery_feat, feats, args):
    """
    MV-IARC: Multi-view Intra-round Adaptive Re-Clustering.

    After the first HDBSCAN + reliability + merge stage, uncertain samples
    (noise + rejected clusters) are re-clustered inside the same incremental round.
    Each secondary cluster is then assigned by a merge / new pseudo-class / reject rule:
        1) if similar to an existing reliable cluster -> merge into it;
        2) else if internally reliable -> create a new pseudo-class cluster;
        3) otherwise -> keep rejected as noise.

    This reduces dependence on the one-shot HDBSCAN result before prototype enrollment.
    """
    labels = np.asarray(labels, dtype=np.int64)
    working = labels.copy()
    accepted_ids = [int(c) for c in accepted_ids if c != -1]

    # Uncertain set = initial noise + clusters not accepted by reliability filtering.
    accepted_set = set(accepted_ids)
    uncertain_mask = (working == -1)
    for cid in np.unique(working):
        if cid == -1:
            continue
        if int(cid) not in accepted_set:
            uncertain_mask |= (working == cid)

    uncertain_idx = np.where(uncertain_mask)[0]

    # Reject uncertain samples by default. IARC only restores samples through merge/new decisions.
    working[uncertain_idx] = -1

    stats = {
        "IARC Uncertain Samples": int(len(uncertain_idx)),
        "IARC Secondary Clusters": 0,
        "IARC Merged Secondary": 0,
        "IARC New Secondary": 0,
        "IARC Rejected Secondary": 0,
    }

    if len(uncertain_idx) < int(args.iarc_min_secondary_size):
        return working.astype(np.int64), accepted_ids, stats

    secondary_labels, secondary_probs = run_hdbscan(
        discovery_feat[uncertain_idx],
        args.iarc_secondary_min_cluster_size,
        args.iarc_secondary_min_samples,
    )
    secondary_cluster_ids = [int(c) for c in np.unique(secondary_labels) if c != -1]
    stats["IARC Secondary Clusters"] = int(len(secondary_cluster_ids))

    if len(secondary_cluster_ids) == 0:
        return working.astype(np.int64), accepted_ids, stats

    # Reliability of secondary clusters in the local uncertain set.
    secondary_accepted, secondary_details = reliability_filter(
        discovery_feat[uncertain_idx],
        secondary_labels,
        secondary_probs,
        args.iarc_min_secondary_size,
        args.reliability_min_prob,
        args.iarc_new_threshold,
    )
    secondary_detail_map = {int(d["cluster_id"]): d for d in secondary_details}
    secondary_accepted = set(int(x) for x in secondary_accepted)

    next_cluster_id = int(np.max(working[working != -1]) + 1) if np.any(working != -1) else 0
    current_accepted = list(accepted_ids)

    for scid in secondary_cluster_ids:
        local_idx = np.where(secondary_labels == scid)[0]
        global_idx = uncertain_idx[local_idx]
        if len(global_idx) < int(args.iarc_min_secondary_size):
            stats["IARC Rejected Secondary"] += 1
            continue

        # Compare secondary cluster with current reliable clusters.
        best_cid, best_score = None, -1.0
        if len(current_accepted) > 0:
            rf_sigma = _estimate_rf_sigma(feats["rf"], working, current_accepted, secondary_indices=global_idx)
            for rcid in current_accepted:
                ref_idx = _cluster_indices(working, rcid)
                if len(ref_idx) == 0:
                    continue
                score, _ = multi_view_cluster_similarity(global_idx, ref_idx, feats, args, rf_sigma)
                if score > best_score:
                    best_score = score
                    best_cid = int(rcid)

        # Decision 1: merge into an existing reliable cluster.
        if best_cid is not None and best_score >= float(args.iarc_merge_threshold):
            working[global_idx] = best_cid
            stats["IARC Merged Secondary"] += 1
            continue

        # Decision 2: create a new pseudo-class if the secondary cluster is reliable itself.
        detail = secondary_detail_map.get(scid, {})
        sec_score = float(detail.get("reliability_score", 0.0))
        if scid in secondary_accepted and sec_score >= float(args.iarc_new_threshold):
            new_id = int(next_cluster_id)
            next_cluster_id += 1
            working[global_idx] = new_id
            current_accepted.append(new_id)
            stats["IARC New Secondary"] += 1
            continue

        # Decision 3: reject as unreliable/noise.
        working[global_idx] = -1
        stats["IARC Rejected Secondary"] += 1

    # Keep only accepted clusters plus IARC-created clusters for enrollment.
    final_accepted = [int(c) for c in current_accepted if np.any(working == c)]
    return working.astype(np.int64), final_accepted, stats


# ============================================================
# Prototype-based incremental evaluation
# ============================================================

def l2norm(z):
    z = np.asarray(z, dtype=np.float32)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return (z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def make_prototypes(features, labels, prototypes_per_class=1, seed=7):
    protos, proto_labels = [], []
    for c in sorted(np.unique(labels).tolist()):
        idx = np.where(labels == c)[0]
        z = features[idx]
        n_proto = min(max(int(prototypes_per_class), 1), len(z))
        if n_proto == 1:
            centers = z.mean(axis=0, keepdims=True)
        else:
            centers = KMeans(n_clusters=n_proto, n_init=10, random_state=seed).fit(z).cluster_centers_
        protos.extend(list(centers))
        proto_labels.extend([int(c)] * len(centers))
    return l2norm(np.stack(protos, axis=0)), np.asarray(proto_labels, dtype=np.int64)


def predict_proto(
    features,
    prototypes,
    proto_labels,
    old_class_count=10,
    old_class_bonus=0.0,
):
    """
    Nearest-prototype classification with optional old-class score calibration.

    Motivation:
        In prototype-based incremental recognition, newly enrolled pseudo-prototypes
        can sometimes attract old-class samples and increase forgetting. A small
        bonus added only to initial known-class prototypes can reduce this bias
        without changing clustering, enrollment, or the feature extractor.

    Args:
        old_class_count:
            Number of initial known classes. In this protocol, Tx0-Tx9 are old classes.
        old_class_bonus:
            Small score bonus beta for initial known-class prototypes.
            Recommended grid: 0.00, 0.02, 0.04, 0.06.
    """
    if prototypes is None or len(prototypes) == 0:
        return np.zeros(len(features), dtype=np.int64)

    sim = l2norm(features) @ l2norm(prototypes).T

    if old_class_bonus > 0:
        old_mask = np.asarray(proto_labels) < int(old_class_count)
        if np.any(old_mask):
            sim[:, old_mask] += float(old_class_bonus)

    return proto_labels[np.argmax(sim, axis=1)]


def acc_on_range(y_true, y_pred, start, end):
    mask = (y_true >= start) & (y_true < end)
    if np.sum(mask) == 0:
        return np.nan
    return float(np.mean(y_true[mask] == y_pred[mask]))


def build_proto_feature_bank(Z_train, X_train, round_Z_list, round_X_list, eval_Z_dict, eval_X_dict, args, cflcg_mode="none"):
    """
    Build normalized feature banks for prototype classification.

    The scalers are fit only on training/discovery data, not evaluation data.
    Multiple held-out evaluation splits are supported:
        eval_initial, eval_r1, eval_r2, eval_r3.
    """
    rf_train = extract_rf_view(X_train, args, cflcg_mode)
    round_rf_list = [extract_rf_view(x, args, cflcg_mode) for x in round_X_list]
    eval_rf_dict = {k: extract_rf_view(x, args, cflcg_mode) for k, x in eval_X_dict.items()}

    # Strict incremental protocol: preprocessing statistics are learned from
    # Day1 known training data only. Future discovery rounds must not influence
    # R1 preprocessing through a jointly fitted scaler.
    deep_scaler = StandardScaler().fit(Z_train)
    rf_scaler = StandardScaler().fit(rf_train)

    def pack(Z, RF):
        d = np.nan_to_num(deep_scaler.transform(Z), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        r = np.nan_to_num(rf_scaler.transform(RF), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return {
            "deep": l2norm(d),
            "rf": l2norm(r),
            "hybrid": l2norm(np.concatenate([d, r], axis=1)),
        }

    bank = {
        "train": pack(Z_train, rf_train),
    }
    for i, (Z, RF) in enumerate(zip(round_Z_list, round_rf_list), start=1):
        bank[f"r{i}"] = pack(Z, RF)
    for key in eval_Z_dict:
        bank[key] = pack(eval_Z_dict[key], eval_rf_dict[key])

    return bank


def enroll_new_prototypes(
    round_features,
    labels,
    accepted_ids,
    next_label,
    reliability_map=None,
    default_reliability=1.0,
    prototypes_per_cluster=1,
    seed=7,
):
    protos, pseudo_labels, proto_reliabilities = [], [], []
    cluster_pseudo_pairs = []
    reliability_map = reliability_map or {}

    for cid in accepted_ids:
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue
        pseudo = int(next_label)
        next_label += 1

        rel = float(reliability_map.get(int(cid), default_reliability))
        rel = float(np.clip(np.nan_to_num(rel, nan=default_reliability, posinf=1.0, neginf=0.0), 0.0, 1.0))

        z = round_features[idx]
        n_proto = min(max(int(prototypes_per_cluster), 1), len(z))
        if n_proto == 1:
            centers = z.mean(axis=0, keepdims=True)
        else:
            centers = KMeans(n_clusters=n_proto, n_init=10, random_state=seed).fit(z).cluster_centers_

        protos.extend(list(centers))
        pseudo_labels.extend([pseudo] * len(centers))
        proto_reliabilities.extend([rel] * len(centers))
        cluster_pseudo_pairs.append((int(cid), pseudo))

    if len(protos) == 0:
        return None, np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32), cluster_pseudo_pairs, next_label

    return (
        l2norm(np.stack(protos, axis=0)),
        np.asarray(pseudo_labels, dtype=np.int64),
        np.asarray(proto_reliabilities, dtype=np.float32),
        cluster_pseudo_pairs,
        next_label,
    )



def evaluate_stage(
    method,
    stage,
    eval_day,
    seen_classes,
    eval_X_name,
    eval_features,
    y_eval,
    prototypes,
    proto_labels,
    pseudo_to_true,
    true_new,
    discovered,
    enrolled,
    initial_known=10,
    round_size=10,
    initial_reference_acc=None,
    old_class_bonus=0.0,
):
    seen_mask = y_eval < seen_classes
    y_true = y_eval[seen_mask]
    pred_raw = predict_proto(
        eval_features[seen_mask],
        prototypes,
        proto_labels,
        old_class_count=initial_known,
        old_class_bonus=old_class_bonus,
    )
    y_pred = np.asarray([pseudo_to_true.get(int(p), int(p)) for p in pred_raw], dtype=np.int64)

    if stage == "Initial":
        old_start, old_end = 0, initial_known
        new_start, new_end = None, None
    else:
        # For After Rk, old classes are all classes seen before the kth round,
        # and new classes are the classes introduced in the kth round.
        try:
            k = int(stage.replace("After R", ""))
        except Exception:
            k = 1
        old_start, old_end = 0, initial_known + (k - 1) * round_size
        new_start = initial_known + (k - 1) * round_size
        new_end = initial_known + k * round_size

    old_acc = acc_on_range(y_true, y_pred, old_start, old_end)
    new_acc = np.nan if new_start is None else acc_on_range(y_true, y_pred, new_start, new_end)
    initial_known_acc = acc_on_range(y_true, y_pred, 0, initial_known)

    if initial_reference_acc is None or np.isnan(initial_known_acc):
        forgetting_rate = 0.0 if stage == "Initial" else np.nan
    else:
        forgetting_rate = float(initial_reference_acc - initial_known_acc)

    return {
        "Method": method,
        "Stage": stage,
        "Eval Day": eval_day,
        "Eval Split": eval_X_name,
        "Seen Classes": int(seen_classes),
        "True New Classes": true_new,
        "Discovered Clusters": discovered,
        "Enrolled Clusters": enrolled,
        "Eval Samples": int(len(y_true)),
        "Overall Acc": float(np.mean(y_true == y_pred)),
        "Old Acc": old_acc,
        "New Acc": new_acc,
        "Initial Known Acc": initial_known_acc,
        "Forgetting Rate": forgetting_rate,
        "Old Class Bonus": float(old_class_bonus),
        "Macro F1": float(f1_score(y_true, y_pred, labels=list(range(seen_classes)), average="macro", zero_division=0)),
    }


def build_per_round_summary(clustering_df, incremental_df):
    """
    Create Table 3: compact per-round summary.

    Compact discovery + incremental summary.  ARI and one-to-one Hungarian
    accuracy are included because Purity alone is inflated by over-clustering.
    """
    # Keep only incremental rounds. Exclude the Initial model row.
    inc = incremental_df[incremental_df["Stage"].astype(str).str.startswith("After R")].copy()
    if inc.empty:
        return pd.DataFrame(columns=[
            "Method", "Round", "Cluster Count", "Incremental Pseudo-label Classes",
            "Cluster Count Error", "Sample Coverage", "Purity", "NMI", "ARI",
            "Hungarian Acc", "Overall Acc", "New Acc", "Forgetting Rate"
        ])

    inc["Round"] = inc["Stage"].astype(str).str.replace("After ", "", regex=False)

    # Cluster Count = raw HDBSCAN discovered clusters before enrollment.
    # Incremental Pseudo-label Classes = final enrolled pseudo-label classes.
    inc = inc[[
        "Method", "Round", "Discovered Clusters", "Enrolled Clusters",
        "Overall Acc", "New Acc", "Forgetting Rate"
    ]].rename(columns={
        "Discovered Clusters": "Cluster Count",
        "Enrolled Clusters": "Incremental Pseudo-label Classes",
    })

    # Discovery metrics come from the same method and round.  Noise-free methods
    # have coverage 1.0; otherwise coverage is 1 - Noise.
    clu = clustering_df[[
        "Method", "Round", "Cluster Count Error", "Noise", "Purity", "NMI", "ARI", "Hungarian Acc"
    ]].copy()
    clu["Sample Coverage"] = 1.0 - pd.to_numeric(clu["Noise"], errors="coerce")
    clu = clu.drop(columns=["Noise"])
    summary = inc.merge(clu, on=["Method", "Round"], how="left")

    summary = summary[[
        "Method", "Round",
        "Cluster Count",
        "Incremental Pseudo-label Classes",
        "Cluster Count Error",
        "Sample Coverage",
        "Purity",
        "NMI",
        "ARI",
        "Hungarian Acc",
        "Overall Acc",
        "New Acc",
        "Forgetting Rate",
    ]]

    method_order = {
        "Deep only": 0,
        "RF only": 1,
        "Graph fusion (raw)": 2,
        "No-drop consolidation (ablation)": 3,
        "MV-ACC": 4,
    }
    round_order = {"R1": 1, "R2": 2, "R3": 3}
    summary["_m"] = summary["Method"].map(method_order).fillna(99)
    summary["_r"] = summary["Round"].map(round_order).fillna(99)
    summary = summary.sort_values(["_m", "_r"]).drop(columns=["_m", "_r"])

    numeric_cols = [
        "Cluster Count",
        "Incremental Pseudo-label Classes",
        "Cluster Count Error",
        "Sample Coverage",
        "Purity",
        "NMI",
        "ARI",
        "Hungarian Acc",
        "Overall Acc",
        "New Acc",
        "Forgetting Rate",
    ]
    for col in numeric_cols:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")

    return summary






# ============================================================
# Representative class-incremental baselines for SOTA comparison
# ============================================================

def _select_exemplars(features, labels, per_class=20):
    """
    Herding-style compact exemplar selection in a fixed feature space.
    For each class, keep samples closest to the class mean.
    """
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    xs, ys = [], []
    for c in sorted(np.unique(labels).tolist()):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        z = features[idx]
        center = z.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(z - center, axis=1)
        keep = idx[np.argsort(dist)[:min(int(per_class), len(idx))]]
        xs.append(features[keep])
        ys.append(np.full(len(keep), int(c), dtype=np.int64))
    if len(xs) == 0:
        return np.empty((0, features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(xs, axis=0).astype(np.float32), np.concatenate(ys, axis=0).astype(np.int64)


def _update_exemplar_memory(memory_X, memory_y, new_X, new_y, per_class=20):
    if memory_X is None or len(memory_X) == 0:
        all_X, all_y = new_X, new_y
    elif new_X is None or len(new_X) == 0:
        all_X, all_y = memory_X, memory_y
    else:
        all_X = np.concatenate([memory_X, new_X], axis=0)
        all_y = np.concatenate([memory_y, new_y], axis=0)
    return _select_exemplars(all_X, all_y, per_class=per_class)


def _make_linear(in_dim, num_outputs, device):
    model = torch.nn.Linear(int(in_dim), int(num_outputs)).to(device)
    return model


def _expand_linear(old_model, in_dim, old_class_ids, new_class_ids, device):
    old_class_ids = list(old_class_ids)
    class_ids = list(old_class_ids)
    for c in new_class_ids:
        if int(c) not in class_ids:
            class_ids.append(int(c))
    new_model = _make_linear(in_dim, len(class_ids), device)
    if old_model is not None and len(old_class_ids) > 0:
        with torch.no_grad():
            old_out = len(old_class_ids)
            new_model.weight[:old_out].copy_(old_model.weight[:old_out])
            new_model.bias[:old_out].copy_(old_model.bias[:old_out])
    return new_model, class_ids


def _train_linear_classifier(
    model,
    train_X,
    train_y,
    class_ids,
    args,
    device,
    teacher_model=None,
    old_output_dim=0,
):
    """
    Train an expandable linear classifier on frozen features.
    Used to implement Ft-CNN/LwF/EEIL-style baselines under the same protocol.
    """
    if train_X is None or len(train_X) == 0:
        return model

    class_to_idx = {int(c): i for i, c in enumerate(class_ids)}
    y_idx = np.asarray([class_to_idx[int(y)] for y in train_y], dtype=np.int64)

    X_t = torch.as_tensor(train_X, dtype=torch.float32)
    y_t = torch.as_tensor(y_idx, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=args.cil_batch_size, shuffle=True, drop_last=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.cil_lr, weight_decay=args.cil_weight_decay)
    T = float(args.cil_distill_temperature)
    kd_weight = float(args.cil_distill_weight)

    if teacher_model is not None:
        teacher_model.eval()

    for _ in range(int(args.cil_epochs)):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            ce = F.cross_entropy(logits, yb)
            loss = ce
            if teacher_model is not None and int(old_output_dim) > 0 and kd_weight > 0:
                with torch.no_grad():
                    old_logits = teacher_model(xb)[:, :old_output_dim]
                new_old_logits = logits[:, :old_output_dim]
                kd = F.kl_div(
                    F.log_softmax(new_old_logits / T, dim=1),
                    F.softmax(old_logits / T, dim=1),
                    reduction="batchmean",
                ) * (T * T)
                loss = ce + kd_weight * kd
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def _predict_linear_classifier(model, features, class_ids, device):
    if model is None or features is None or len(features) == 0:
        return np.zeros(0, dtype=np.int64)
    model.eval()
    X = torch.as_tensor(features, dtype=torch.float32, device=device)
    logits = model(X).detach().cpu().numpy()
    idx = np.argmax(logits, axis=1)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    return class_ids[idx]


def build_posthoc_pseudo_to_true(y_true, cluster_labels, cluster_pseudo_pairs):
    """Build a strict one-to-one label alignment for reporting only.

    Cluster identifiers are arbitrary, so a post-hoc alignment is required to
    report identification accuracy.  A many-to-one majority mapping makes
    severe over-clustering look artificially good because several pseudo
    classes can all be credited as the same transmitter.  This implementation
    instead uses a Hungarian one-to-one assignment.  Extra pseudo classes are
    mapped to negative sentinel labels and therefore count as errors.

    Ground truth is never used to create prototypes, train a classifier,
    choose clustering parameters, merge clusters, or reassign samples.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    pairs = [(int(cid), int(pseudo)) for cid, pseudo in cluster_pseudo_pairs]
    mapping = {pseudo: -(1_000_000 + pseudo) for _, pseudo in pairs}
    if not pairs or len(y_true) == 0:
        return mapping

    true_ids = np.unique(y_true)
    contingency = np.zeros((len(true_ids), len(pairs)), dtype=np.int64)
    true_pos = {int(v): i for i, v in enumerate(true_ids)}
    for j, (cid, _) in enumerate(pairs):
        idx = np.where(cluster_labels == cid)[0]
        for value, count in zip(*np.unique(y_true[idx], return_counts=True)):
            contingency[true_pos[int(value)], j] = int(count)

    row_ind, col_ind = linear_sum_assignment(-contingency)
    for i, j in zip(row_ind, col_ind):
        _, pseudo = pairs[int(j)]
        mapping[pseudo] = int(true_ids[int(i)])
    return mapping


def _make_pseudo_labeled_round(round_features, labels, enrolled_ids, next_label):
    xs, ys = [], []
    cluster_pseudo_pairs = []
    for cid in enrolled_ids:
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue
        pseudo = int(next_label)
        next_label += 1
        cluster_pseudo_pairs.append((int(cid), pseudo))
        xs.append(round_features[idx])
        ys.append(np.full(len(idx), pseudo, dtype=np.int64))
    if len(xs) == 0:
        return np.empty((0, round_features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64), cluster_pseudo_pairs, next_label
    return np.concatenate(xs, axis=0).astype(np.float32), np.concatenate(ys, axis=0).astype(np.int64), cluster_pseudo_pairs, next_label


def _align_refined_clusters_to_pseudo(
    reference_features,
    reference_pseudo_y,
    refined_features,
    refined_cluster_y,
):
    """将 Student 二次发现的簇无监督对齐回第一次注册的伪类编号。

    聚类标签本身没有固定语义。第一次 CIL 更新后，如果直接把 Student
    新发现的簇编号拿去训练，编号变化会破坏已有分类头；这里仅使用当前轮
    discovery/train 侧的类中心做余弦相似度 Hungarian 匹配，不读取 held-out
    eval 或未知真实标签。
    """

    reference_features = np.asarray(reference_features, dtype=np.float32)
    reference_pseudo_y = np.asarray(reference_pseudo_y, dtype=np.int64)
    refined_features = np.asarray(refined_features, dtype=np.float32)
    refined_cluster_y = np.asarray(refined_cluster_y, dtype=np.int64)
    reference_ids = sorted(np.unique(reference_pseudo_y).tolist())
    refined_ids = sorted(np.unique(refined_cluster_y).tolist())
    if not reference_ids or not refined_ids:
        return reference_pseudo_y.copy()

    reference_centers = np.stack(
        [reference_features[reference_pseudo_y == int(label)].mean(axis=0) for label in reference_ids]
    )
    refined_centers = np.stack(
        [refined_features[refined_cluster_y == int(label)].mean(axis=0) for label in refined_ids]
    )
    similarity = cosine_similarity(reference_centers, refined_centers)
    row_ind, col_ind = linear_sum_assignment(-similarity)
    cluster_to_pseudo = {
        int(refined_ids[col]): int(reference_ids[row])
        for row, col in zip(row_ind.tolist(), col_ind.tolist())
    }
    fallback = iter(reference_ids)
    for cluster_id in refined_ids:
        cluster_to_pseudo.setdefault(int(cluster_id), int(next(fallback, reference_ids[0])))
    return np.asarray(
        [cluster_to_pseudo[int(cluster_id)] for cluster_id in refined_cluster_y],
        dtype=np.int64,
    )


def _evaluate_raw_predictions(
    method,
    stage,
    eval_day,
    seen_classes,
    eval_X_name,
    y_eval,
    pred_raw,
    pseudo_to_true,
    true_new,
    discovered,
    enrolled,
    initial_known=10,
    round_size=10,
    initial_reference_acc=None,
):
    seen_mask = y_eval < seen_classes
    y_true = y_eval[seen_mask]
    pred_raw = np.asarray(pred_raw, dtype=np.int64)
    y_pred = np.asarray([pseudo_to_true.get(int(p), int(p)) for p in pred_raw], dtype=np.int64)

    if stage == "Initial":
        old_start, old_end = 0, initial_known
        new_start, new_end = None, None
    else:
        try:
            k = int(stage.replace("After R", ""))
        except Exception:
            k = 1
        old_start, old_end = 0, initial_known + (k - 1) * round_size
        new_start = initial_known + (k - 1) * round_size
        new_end = initial_known + k * round_size

    old_acc = acc_on_range(y_true, y_pred, old_start, old_end)
    new_acc = np.nan if new_start is None else acc_on_range(y_true, y_pred, new_start, new_end)
    initial_known_acc = acc_on_range(y_true, y_pred, 0, initial_known)
    if initial_reference_acc is None or np.isnan(initial_known_acc):
        forgetting_rate = 0.0 if stage == "Initial" else np.nan
    else:
        forgetting_rate = float(initial_reference_acc - initial_known_acc)

    return {
        "Method": method,
        "Stage": stage,
        "Eval Day": eval_day,
        "Eval Split": eval_X_name,
        "Seen Classes": int(seen_classes),
        "True New Classes": true_new,
        "Discovered Clusters": discovered,
        "Enrolled Clusters": enrolled,
        "Eval Samples": int(len(y_true)),
        "Overall Acc": float(np.mean(y_true == y_pred)),
        "Old Acc": old_acc,
        "New Acc": new_acc,
        "Initial Known Acc": initial_known_acc,
        "Forgetting Rate": forgetting_rate,
        "Macro F1": float(f1_score(y_true, y_pred, labels=list(range(seen_classes)), average="macro", zero_division=0)),
    }


def _majority_int(values):
    """对一个 recording 内的样本预测做稳定多数投票。

    LoRa 的 held-out eval 由同一 transmission/recording 下多个 symbol 片段组成。
    这里不读取任何评估真值做选择，只把同一可观测 recording_id 的逐 symbol
    预测聚合成一个预测；若票数并列，选择数值较小的类别，保证同 seed 可复现。
    """
    values = np.asarray(values, dtype=np.int64)
    if len(values) == 0:
        raise ValueError("majority vote requires at least one value.")
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[np.argmax(counts)])


def _evaluate_recording_level_predictions(
    method,
    stage,
    eval_day,
    seen_classes,
    eval_X_name,
    y_eval,
    pred_raw,
    pseudo_to_true,
    recording_id,
    true_new,
    discovered,
    enrolled,
    initial_known=10,
    round_size=10,
    initial_reference_acc=None,
):
    """按 LoRa recording_id 汇总 held-out eval 预测并计算诊断指标。

    这是评估粒度诊断，不改变训练、发现、伪标签注册或模型选择。分组键来自
    LoRa 数据切分中可观测的 recording_id；真实标签只在聚合完成后用于离线
    统计指标，因此仍然遵守 held-out evaluation 不参与训练/调参的边界。
    """
    if recording_id is None:
        return None

    y_eval = np.asarray(y_eval, dtype=np.int64)
    pred_raw = np.asarray(pred_raw, dtype=np.int64)
    groups = np.asarray(recording_id)
    if groups.shape != y_eval.shape or pred_raw.shape != y_eval.shape:
        raise ValueError(
            "recording-level eval shape mismatch: "
            f"recording_id={groups.shape}, y={y_eval.shape}, pred={pred_raw.shape}"
        )

    seen_mask = y_eval < int(seen_classes)
    if not np.any(seen_mask):
        return None

    y_seen = y_eval[seen_mask]
    pred_seen = np.asarray([pseudo_to_true.get(int(p), int(p)) for p in pred_raw[seen_mask]], dtype=np.int64)
    group_seen = groups[seen_mask]

    group_true, group_pred = [], []
    for gid in sorted(np.unique(group_seen).tolist()):
        idx = group_seen == gid
        group_true.append(_majority_int(y_seen[idx]))
        group_pred.append(_majority_int(pred_seen[idx]))

    row = _evaluate_raw_predictions(
        method,
        stage,
        eval_day,
        seen_classes,
        eval_X_name,
        np.asarray(group_true, dtype=np.int64),
        np.asarray(group_pred, dtype=np.int64),
        {},
        true_new,
        discovered,
        enrolled,
        initial_known,
        round_size,
        initial_reference_acc,
    )
    row["Eval Granularity"] = "recording_majority_vote"
    row["Source Eval Samples"] = int(len(y_seen))
    row["Recording Groups"] = int(len(group_true))
    return row


# ============================================================
# End-to-end pseudo-label class-incremental learning (MV-ACC-CIL)
# ============================================================

def _expand_closedset_classifier(
    old_model,
    new_out_dim,
    device,
    new_class_feature_means=None,
    new_imprint_scale=1.0,
    new_imprint_bias=0.0,
):
    """Expand the neural classifier while copying every old output weight."""
    old_out = int(old_model.classifier.out_features)
    if int(new_out_dim) < old_out:
        raise ValueError("The incremental classifier cannot shrink.")
    student = copy.deepcopy(old_model).to(device)
    if int(new_out_dim) == old_out:
        return student
    new_head = torch.nn.Linear(int(old_model.feat_dim), int(new_out_dim)).to(device)
    with torch.no_grad():
        new_head.weight[:old_out].copy_(old_model.classifier.weight)
        new_head.bias[:old_out].copy_(old_model.classifier.bias)
        # Classifier-weight imprinting 只作为新类 head 的初始化，不改变最终评估口径。
        # Stage51 允许显式放大新类初始 weight/bias，用来验证 LoRa seed31 的
        # 低 New 是否来自“新类刚注册时 logits 过弱”。该机制只使用当前轮
        # discovery 的伪类特征均值，不读取 held-out eval 或未知真值。
        if new_class_feature_means is not None:
            means = torch.as_tensor(new_class_feature_means, dtype=new_head.weight.dtype, device=device)
            n = min(len(means), int(new_out_dim) - old_out)
            old_scale = old_model.classifier.weight.norm(dim=1).mean().clamp_min(1e-8)
            means = F.normalize(means[:n], dim=1) * old_scale * float(new_imprint_scale)
            new_head.weight[old_out:old_out + n].copy_(means)
            new_head.bias[old_out:old_out + n].fill_(float(new_imprint_bias))
    student.classifier = new_head
    student.num_known_classes = int(new_out_dim)
    return student


def _update_iq_memory(memory_x, memory_y, new_x, new_y, per_class, seed):
    """Balanced raw-IQ replay memory; no frozen features or prototype classifier."""
    if memory_x is None or len(memory_x) == 0:
        all_x, all_y = np.asarray(new_x), np.asarray(new_y)
    else:
        all_x = np.concatenate([memory_x, new_x], axis=0)
        all_y = np.concatenate([memory_y, new_y], axis=0)
    rng = np.random.default_rng(int(seed))
    keep = []
    for c in sorted(np.unique(all_y).tolist()):
        idx = np.where(all_y == c)[0]
        take = min(int(per_class), len(idx))
        keep.extend(rng.choice(idx, size=take, replace=False).tolist())
    keep = np.asarray(sorted(keep), dtype=np.int64)
    return all_x[keep].astype(np.float32), all_y[keep].astype(np.int64)


def _supcon_incremental(features, labels, temperature=0.2):
    """Supervised contrastive loss on a selected, label-reliable mini-batch."""
    if len(labels) < 2:
        return features.new_tensor(0.0)
    z = F.normalize(features, dim=1)
    logits = (z @ z.T) / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(len(labels), device=features.device, dtype=torch.bool)
    same = labels[:, None].eq(labels[None, :]) & (~eye)
    denom_mask = ~eye
    log_prob = logits - torch.log((torch.exp(logits) * denom_mask).sum(dim=1, keepdim=True) + 1e-12)
    positives = same.sum(dim=1)
    valid = positives > 0
    if not torch.any(valid):
        return features.new_tensor(0.0)
    return -(log_prob * same.float()).sum(dim=1)[valid].div(positives[valid].float()).mean()


def _feature_view_consistency_loss(features, augmented_features):
    """计算原始 IQ 与轻度增强 IQ 的单位特征一致性损失。

    该损失不需要额外标签，也不把不同类别样本强行拉到一起；它只要求同一
    个样本在轻微幅度、相位、时间偏移和噪声扰动下保持方向一致。这样可以
    给跨天采集造成的低阶域变化留出适应空间，同时避免把 Teacher 的旧域
    特征逐样本硬复制到当前轮新类。
    """
    if features.numel() == 0 or augmented_features.numel() == 0:
        return features.sum() * 0.0
    if features.shape != augmented_features.shape:
        raise ValueError(
            "Feature-view consistency requires matching feature shapes, got "
            f"{tuple(features.shape)} and {tuple(augmented_features.shape)}."
        )
    cosine = F.cosine_similarity(
        F.normalize(features, dim=1),
        F.normalize(augmented_features, dim=1),
        dim=1,
    )
    return (1.0 - cosine).mean()


def _feature_domain_alignment_loss(source_features, target_features):
    """对齐当前轮与 replay 的无标签特征分布，缓解跨天采集漂移。

    先比较单位化特征的一阶均值，再比较二阶协方差。两组样本都来自当前
    在线训练可见的数据，不读取 held-out evaluation，也不需要未知类真值。
    """
    if source_features.numel() == 0 or target_features.numel() == 0:
        return source_features.sum() * 0.0
    source = F.normalize(source_features, dim=1)
    target = F.normalize(target_features, dim=1)
    mean_loss = (source.mean(dim=0) - target.mean(dim=0)).pow(2).mean()
    if source.shape[0] < 2 or target.shape[0] < 2:
        return mean_loss
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    source_cov = source_centered.T @ source_centered / float(source.shape[0] - 1)
    target_cov = target_centered.T @ target_centered / float(target.shape[0] - 1)
    covariance_loss = (source_cov - target_cov).pow(2).mean()
    return mean_loss + covariance_loss


def _compute_pseudo_sample_weights(
    labels,
    raw_probabilities,
    reliability_map=None,
    floor=0.20,
    use_cluster_reliability=False,
):
    """把逐样本概率和无标签簇可靠性合成为伪标签训练权重。

    ``raw_probabilities`` 通常来自 HDBSCAN；``reliability_map`` 来自当前轮
    discovery 的簇紧凑度、簇间分离度和概率综合评分。两者都只使用
    discovery 样本，不依赖当前轮或 held-out evaluation 的真实标签。GPCC
    没有噪声簇时默认可靠性为 1，不会改变它的默认训练路径。
    """
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(raw_probabilities, dtype=np.float32)
    if len(labels) != len(probabilities):
        raise ValueError(
            "Pseudo-label weights require labels and probabilities with equal length."
        )
    floor = float(np.clip(floor, 0.0, 1.0))
    weights = np.maximum(
        floor,
        np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0),
    )
    if not use_cluster_reliability:
        return weights.astype(np.float32)
    reliability_map = reliability_map or {}
    cluster_reliability = np.asarray(
        [
            float(
                np.clip(
                    np.nan_to_num(
                        reliability_map.get(int(label), 1.0),
                        nan=1.0,
                        posinf=1.0,
                        neginf=0.0,
                    ),
                    0.0,
                    1.0,
                )
            )
            for label in labels
        ],
        dtype=np.float32,
    )
    return np.maximum(floor, weights * cluster_reliability).astype(np.float32)


def _select_pseudo_registration_mask(pseudo_y, confidence, top_fraction, min_per_class):
    """选择只用于新类注册和 replay 的高置信样本。

    当前轮完整训练仍保留全部 discovery 样本，避免高置信筛选把新类训练
    数据量压得过小。这个 mask 只控制两处容易被错误伪标签污染的位置：
    新分类头的 feature imprint 和下一轮 replay memory。选择过程只依赖
    无标签聚类置信度及伪类内部排序，不读取未知真实标签。
    """
    pseudo_y = np.asarray(pseudo_y, dtype=np.int64)
    confidence = np.asarray(confidence, dtype=np.float32)
    if pseudo_y.size == 0:
        return np.zeros(0, dtype=bool)
    if float(top_fraction) >= 0.999:
        return np.ones(pseudo_y.size, dtype=bool)
    if confidence.shape != pseudo_y.shape:
        confidence = np.ones(pseudo_y.size, dtype=np.float32)
    keep = np.zeros(pseudo_y.size, dtype=bool)
    fraction = float(np.clip(top_fraction, 0.01, 1.0))
    minimum = max(1, int(min_per_class))
    for class_id in sorted(np.unique(pseudo_y).tolist()):
        indices = np.where(pseudo_y == int(class_id))[0]
        if indices.size == 0:
            continue
        take = min(
            indices.size,
            max(minimum, int(np.ceil(indices.size * fraction))),
        )
        order = np.argsort(-confidence[indices], kind="mergesort")
        keep[indices[order[:take]]] = True
    return keep


def _select_refined_consensus_training_mask(refined_pseudo_y, logits, top_fraction, min_per_class):
    """为二次 discovery refinement 选择高置信且与 Student 预测一致的样本。

    Stage56 只作用于 ``enable_joint_discovery_refinement`` 的第二次 CIL。第一次
    discovery 仍按原协议固定注册所有伪类；第一次 CIL 后，Student 已经给当前轮
    discovery 样本形成了一个网络分类判断。这里要求样本满足两个训练期可见条件：

    1. 二次聚类对齐后的伪标签与 Student 最大 logit 预测一致；
    2. 在该伪类内部属于较高置信样本。

    该函数不读取未知真实标签或 held-out eval，只使用当前轮 discovery 样本的
    伪标签和 Student logits。若某个伪类完全没有一致样本，则保留该类最高置信
    的一个样本，避免二次训练把已注册新类整列饿死。
    """

    labels = np.asarray(refined_pseudo_y, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)
    if labels.size == 0:
        return np.zeros(0, dtype=bool), {
            "Refined Filter Kept": 0,
            "Refined Filter Total": 0,
            "Refined Filter Keep Rate": 0.0,
            "Refined Filter Agreement Rate": 0.0,
            "Refined Filter Mean Confidence": 0.0,
        }
    if logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Refined consensus filter expects logits shape (N,C), got {logits.shape} for {labels.shape} labels."
        )
    if float(top_fraction) >= 0.999:
        keep = np.ones(labels.size, dtype=bool)
        predictions = np.argmax(logits, axis=1)
        agreement = predictions == labels
        return keep, {
            "Refined Filter Kept": int(labels.size),
            "Refined Filter Total": int(labels.size),
            "Refined Filter Keep Rate": 1.0,
            "Refined Filter Agreement Rate": float(np.mean(agreement)),
            "Refined Filter Mean Confidence": 1.0,
        }

    stable_logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(stable_logits)
    probs = exp_logits / np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-12)
    predictions = np.argmax(probs, axis=1).astype(np.int64)
    label_indices = np.clip(labels, 0, probs.shape[1] - 1)
    assigned_confidence = probs[np.arange(labels.size), label_indices].astype(np.float32)
    agreement = predictions == labels

    keep = np.zeros(labels.size, dtype=bool)
    fraction = float(np.clip(top_fraction, 0.01, 1.0))
    minimum = max(1, int(min_per_class))
    fallback_classes = 0
    for class_id in sorted(np.unique(labels).tolist()):
        class_indices = np.where(labels == int(class_id))[0]
        if class_indices.size == 0:
            continue
        target = min(class_indices.size, max(minimum, int(np.ceil(class_indices.size * fraction))))
        agreed = class_indices[agreement[class_indices]]
        if agreed.size > 0:
            order = np.argsort(-assigned_confidence[agreed], kind="mergesort")
            chosen = agreed[order[: min(target, agreed.size)]]
        else:
            fallback_classes += 1
            order = np.argsort(-assigned_confidence[class_indices], kind="mergesort")
            chosen = class_indices[order[:1]]
        keep[chosen] = True

    if not np.any(keep):
        keep[np.argmax(assigned_confidence)] = True
    kept_conf = assigned_confidence[keep]
    diagnostics = {
        "Refined Filter Kept": int(np.sum(keep)),
        "Refined Filter Total": int(labels.size),
        "Refined Filter Keep Rate": float(np.mean(keep)),
        "Refined Filter Agreement Rate": float(np.mean(agreement)),
        "Refined Filter Mean Confidence": float(np.mean(kept_conf)) if kept_conf.size else 0.0,
        "Refined Filter Fallback Classes": int(fallback_classes),
    }
    return keep, diagnostics


def _build_current_pseudo_prototypes(student, current_x, current_y, device):
    """用当前轮开始时的 Student 特征建立固定伪类原型。

    原型只由当前 discovery 样本和伪标签构成，不读取未知真实标签；训练中
    用它约束 Student 的新类角度归属，避免分类头只靠交叉熵记住局部伪标签。
    """
    current_x = np.asarray(current_x, dtype=np.float32)
    current_y = np.asarray(current_y, dtype=np.int64)
    if current_x.size == 0:
        return None, None
    loader = DataLoader(
        TensorDataset(torch.as_tensor(current_x, dtype=torch.float32)),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )
    features = []
    student.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            feat, _ = student(batch_x.to(device))
            features.append(feat)
    all_features = torch.cat(features, dim=0)
    prototype_count = int(student.classifier.out_features)
    prototypes = torch.zeros(
        (prototype_count, all_features.shape[1]),
        dtype=all_features.dtype,
        device=device,
    )
    valid = torch.zeros(prototype_count, dtype=torch.bool, device=device)
    labels_device = torch.as_tensor(current_y, dtype=torch.long, device=device)
    for class_id in sorted(np.unique(current_y).tolist()):
        if 0 <= int(class_id) < prototype_count:
            mask = labels_device == int(class_id)
            if torch.any(mask):
                prototypes[int(class_id)] = F.normalize(
                    all_features[mask].mean(dim=0), dim=0
                )
                valid[int(class_id)] = True
    return prototypes, valid


def _prototype_assignment_loss(features, labels, prototypes, valid, temperature=0.2):
    """计算当前轮样本到固定伪类原型的角度分类损失。"""
    if prototypes is None or valid is None or not torch.any(valid):
        return features.sum() * 0.0
    normalized = F.normalize(features, dim=1)
    logits = normalized @ F.normalize(prototypes, dim=1).T
    logits = logits / max(float(temperature), 1e-4)
    logits = logits.masked_fill(~valid.unsqueeze(0), -1e4)
    usable = valid[labels.clamp(min=0, max=valid.numel() - 1)]
    if not torch.any(usable):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[usable], labels[usable])


def _recording_logit_consistency_loss(logits, labels, recording_ids, temperature=2.0):
    """约束同一 LoRa recording 内、同一伪类样本的预测分布保持一致。

    LoRa strict split 中一个 recording 会被切成多个 symbol 片段。Stage48 的
    recording-level 诊断高于 symbol-level，说明同一 transmission 内存在可利用
    的稳定结构。这里仅使用训练期 current discovery 的可观测 recording_id 和
    已有伪标签，不读取 held-out eval，也不把不同伪类强行拉到一起；同一
    recording 若被聚类成多个伪类，则按 ``recording_id + pseudo label`` 分组。
    """
    if recording_ids is None or logits.numel() == 0 or logits.shape[0] <= 1:
        return logits.sum() * 0.0
    if logits.shape[0] != recording_ids.shape[0] or logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "Recording consistency requires logits, labels and recording_ids with equal batch length."
        )

    temperature = max(float(temperature), 1e-4)
    log_probs = F.log_softmax(logits / temperature, dim=1)
    probs = log_probs.exp()
    losses = []
    for recording_value in torch.unique(recording_ids).tolist():
        recording_mask = recording_ids == int(recording_value)
        if int(recording_mask.sum().item()) < 2:
            continue
        for label_value in torch.unique(labels[recording_mask]).tolist():
            group_mask = recording_mask & (labels == int(label_value))
            if int(group_mask.sum().item()) < 2:
                continue
            target = probs[group_mask].mean(dim=0).detach()
            losses.append(
                F.kl_div(
                    log_probs[group_mask],
                    target.expand(int(group_mask.sum().item()), -1),
                    reduction="batchmean",
                )
                * (temperature * temperature)
            )
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _recording_mean_logit_ce_loss(logits, labels, sample_weights, recording_ids):
    """对 LoRa recording 内同伪类 symbol 的平均 logits 直接做交叉熵。

    Stage53 的一致性项只要求同一 recording 内预测分布彼此接近，但没有改变
    当前轮伪标签监督的粒度。这里把同一 ``recording_id + pseudo label`` 的
    多个 symbol logits 先求均值，再对聚合后的 recording-level 表示做 CE。
    它仍然只使用训练期可见的 current discovery 伪标签和 recording_id，不读取
    held-out eval，也不假设同一 recording 中不同伪类应该被合并。
    """
    if recording_ids is None or logits.numel() == 0:
        return logits.sum() * 0.0
    if logits.shape[0] != recording_ids.shape[0] or logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "Recording mean-logit CE requires logits, labels and recording_ids with equal batch length."
        )
    if sample_weights.shape[0] != logits.shape[0]:
        raise ValueError("Recording mean-logit CE requires one sample weight per current sample.")

    group_logits = []
    group_labels = []
    group_weights = []
    for recording_value in torch.unique(recording_ids).tolist():
        recording_mask = recording_ids == int(recording_value)
        for label_value in torch.unique(labels[recording_mask]).tolist():
            group_mask = recording_mask & (labels == int(label_value))
            if not torch.any(group_mask):
                continue
            group_logits.append(logits[group_mask].mean(dim=0))
            group_labels.append(int(label_value))
            group_weights.append(sample_weights[group_mask].mean())
    if not group_logits:
        return logits.sum() * 0.0
    stacked_logits = torch.stack(group_logits, dim=0)
    stacked_labels = torch.as_tensor(group_labels, dtype=torch.long, device=logits.device)
    stacked_weights = torch.stack(group_weights).to(logits.device).clamp_min(1e-6)
    per_group = F.cross_entropy(stacked_logits, stacked_labels, reduction="none")
    return (per_group * stacked_weights).sum() / (stacked_weights.sum() + 1e-8)


def _train_end_to_end_cil(
    student,
    teacher,
    current_x,
    current_y,
    current_w,
    memory_x,
    memory_y,
    old_out_dim,
    args,
    device,
    current_recording_id=None,
    old_day_calibration_x=None,
    old_day_calibration_y=None,
):
    """Head warm-up followed by last-block backbone adaptation on raw IQ samples.

    ``radcil_aug_consistency_weight`` 开启后，当前轮 discovery 和 replay 样本
    会各自生成一个只用于训练的增强视图，并加入同一样本的特征方向一致性
    损失。增强视图不会进入聚类、验证或 held-out 评估，因此不改变严格
    数据边界；权重为 0 时完全保持历史训练路径。
    """
    current_x = np.asarray(current_x, dtype=np.float32)
    current_y = np.asarray(current_y, dtype=np.int64)
    current_w = np.asarray(current_w, dtype=np.float32)
    memory_x = np.asarray(memory_x, dtype=np.float32)
    memory_y = np.asarray(memory_y, dtype=np.int64)
    calibration_loader = None
    if old_day_calibration_x is not None or old_day_calibration_y is not None:
        if old_day_calibration_x is None or old_day_calibration_y is None:
            raise ValueError("Old-day calibration inputs must be provided together.")
        old_day_calibration_x = np.asarray(old_day_calibration_x, dtype=np.float32)
        old_day_calibration_y = np.asarray(old_day_calibration_y, dtype=np.int64)
        if len(old_day_calibration_x) == 0:
            raise ValueError("Old-day calibration data must not be empty when enabled.")
        if len(old_day_calibration_x) != len(old_day_calibration_y):
            raise ValueError(
                "Old-day calibration arrays have inconsistent lengths: "
                f"{len(old_day_calibration_x)} vs {len(old_day_calibration_y)}"
            )
        if np.any(old_day_calibration_y < 0) or np.any(old_day_calibration_y >= int(old_out_dim)):
            raise ValueError(
                "Old-day calibration labels must refer to classes already registered "
                f"in the current round: [0, {int(old_out_dim) - 1}]"
            )
        calibration_loader = DataLoader(
            TensorDataset(
                torch.as_tensor(old_day_calibration_x, dtype=torch.float32),
                torch.as_tensor(old_day_calibration_y, dtype=torch.long),
            ),
            batch_size=max(2, int(args.lora_old_day_joint_cil_batch_size)),
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
    recording_tensor = None
    if current_recording_id is not None:
        current_recording_id = np.asarray(current_recording_id, dtype=np.int64)
        if current_recording_id.shape != current_y.shape:
            raise ValueError(
                f"current_recording_id shape mismatch: {current_recording_id.shape} vs {current_y.shape}"
            )
        recording_tensor = torch.as_tensor(current_recording_id, dtype=torch.long)
    # 默认保持旧实现：当前伪标签 batch 和 replay batch 使用相同大小。
    # 若显式设置 old:new batch 比例，则只调整 replay loader 的 batch size，
    # 便于独立验证“采样配比”和“replay loss 权重”的贡献。
    current_batch_size = int(args.incremental_batch_size)
    if float(args.radcil_old_new_batch_ratio) > 0:
        memory_batch_size = max(1, int(round(current_batch_size * float(args.radcil_old_new_batch_ratio))))
    else:
        memory_batch_size = current_batch_size
    if recording_tensor is None:
        cur_dataset = TensorDataset(torch.as_tensor(current_x), torch.as_tensor(current_y), torch.as_tensor(current_w))
    else:
        cur_dataset = TensorDataset(
            torch.as_tensor(current_x),
            torch.as_tensor(current_y),
            torch.as_tensor(current_w),
            recording_tensor,
        )
    cur_loader = DataLoader(
        cur_dataset,
        batch_size=current_batch_size, shuffle=True, drop_last=False, num_workers=0,
    )
    mem_loader = DataLoader(
        TensorDataset(torch.as_tensor(memory_x), torch.as_tensor(memory_y)),
        batch_size=memory_batch_size, shuffle=True, drop_last=False, num_workers=0,
    )
    teacher = copy.deepcopy(teacher).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    current_prototypes = None
    current_prototype_valid = None
    if (
        float(args.radcil_new_prototype_weight) > 0
        or float(args.radcil_pseudo_aug_consistency_weight) > 0
    ):
        current_prototypes, current_prototype_valid = _build_current_pseudo_prototypes(
            student,
            current_x,
            current_y,
            device,
        )

    # 类中心锚定使用本轮训练开始前的 Teacher 和旧类 replay 构建固定原型。
    # 与逐样本特征蒸馏不同，它只约束旧类的类级中心，不要求 Student 复制每个
    # 样本的瞬时特征，因此给跨天域偏移保留适应空间。当前轮新伪类尚未进入
    # memory，且标签边界由 old_out_dim 给出，不读取任何 held-out 真值。
    anchor_prototypes = None
    anchor_valid = None
    if float(args.radcil_prototype_anchor_weight) > 0:
        teacher_features = []
        anchor_loader = DataLoader(
            TensorDataset(torch.as_tensor(memory_x, dtype=torch.float32)),
            batch_size=int(args.test_batch_size), shuffle=False, num_workers=0,
        )
        with torch.no_grad():
            for (anchor_x,) in anchor_loader:
                anchor_feat, _ = teacher(anchor_x.to(device))
                teacher_features.append(anchor_feat)
        all_teacher_features = torch.cat(teacher_features, dim=0)
        memory_y_device = torch.as_tensor(memory_y, dtype=torch.long, device=device)
        anchor_prototypes = torch.zeros(
            (int(old_out_dim), all_teacher_features.shape[1]),
            dtype=all_teacher_features.dtype, device=device,
        )
        anchor_valid = torch.zeros(int(old_out_dim), dtype=torch.bool, device=device)
        for class_id in range(int(old_out_dim)):
            class_mask = memory_y_device == class_id
            if torch.any(class_mask):
                anchor_prototypes[class_id] = F.normalize(
                    all_teacher_features[class_mask].mean(dim=0), dim=0
                )
                anchor_valid[class_id] = True

    def trainable_backbone_parameters():
        """按 RADCIL 消融参数选择需要解冻的骨干末端范围。"""
        scope = str(args.radcil_unfreeze_scope).lower()
        if hasattr(student.backbone, "trainable_scope_parameters"):
            return list(student.backbone.trainable_scope_parameters(scope))
        if scope == "none":
            return []
        if scope == "fc":
            return list(student.backbone.fc.parameters())
        if scope == "layer3":
            return list(student.backbone.layer3.parameters())
        if scope == "tail":
            return list(student.backbone.layer3.parameters()) + list(student.backbone.fc.parameters())
        if scope == "layer2_tail":
            return (
                list(student.backbone.layer2.parameters())
                + list(student.backbone.layer3.parameters())
                + list(student.backbone.fc.parameters())
            )
        raise ValueError(f"Unsupported --radcil_unfreeze_scope: {args.radcil_unfreeze_scope}")

    def scheduled_kd_weight(epoch_index, total_epochs):
        """计算当前 epoch 的 KD 权重，默认 constant 与旧实现完全一致。"""
        base = float(args.cil_kd_weight)
        schedule = str(args.radcil_kd_schedule).lower()
        if base <= 0:
            return 0.0
        if schedule == "constant":
            return base
        progress = 0.0 if total_epochs <= 1 else float(epoch_index) / float(total_epochs - 1)
        if schedule == "cosine":
            return base * 0.5 * (1.0 + float(np.cos(np.pi * progress)))
        if schedule == "linear_decay":
            return base * (1.0 - progress)
        raise ValueError(f"Unsupported --radcil_kd_schedule: {args.radcil_kd_schedule}")

    def configure(stage):
        for p in student.backbone.parameters():
            p.requires_grad_(False)
        for p in student.classifier.parameters():
            p.requires_grad_(True)
        backbone_params = []
        if stage == "joint":
            backbone_params = trainable_backbone_parameters()
            for p in backbone_params:
                p.requires_grad_(True)
        groups = [{"params": student.classifier.parameters(), "lr": float(args.cil_classifier_lr)}]
        if stage == "joint" and len(backbone_params) > 0:
            groups.append({"params": backbone_params, "lr": float(args.cil_backbone_lr)})
        return torch.optim.AdamW(groups, weight_decay=float(args.cil_weight_decay))

    for stage, epochs in (("head", int(args.cil_head_warmup_epochs)), ("joint", int(args.cil_joint_epochs))):
        if epochs <= 0:
            continue
        opt = configure(stage)
        for epoch_index in range(epochs):
            student.train()
            mem_iter = itertools.cycle(mem_loader)
            calibration_iter = (
                itertools.cycle(calibration_loader)
                if calibration_loader is not None
                else None
            )
            for current_batch in cur_loader:
                if recording_tensor is None:
                    xc, yc, wc = current_batch
                    rc = None
                else:
                    xc, yc, wc, rc = current_batch
                xm, ym = next(mem_iter)
                xc, yc, wc = xc.to(device), yc.to(device), wc.to(device)
                rc = rc.to(device) if rc is not None else None
                xm, ym = xm.to(device), ym.to(device)
                x = torch.cat([xc, xm], dim=0)
                feat, logits = student(x)
                logits_c, logits_m = logits[:len(xc)], logits[len(xc):]
                if calibration_iter is not None:
                    xcal, ycal = next(calibration_iter)
                    xcal, ycal = xcal.to(device), ycal.to(device)
                    calibration_feat, calibration_logits = student(xcal)
                    with torch.no_grad():
                        teacher_calibration_feat, _ = teacher(xcal)
                    old_day_calibration_ce = F.cross_entropy(calibration_logits, ycal)
                    old_day_calibration_feature = F.mse_loss(
                        F.normalize(calibration_feat, dim=1),
                        F.normalize(teacher_calibration_feat, dim=1),
                    )
                else:
                    old_day_calibration_ce = logits_m.sum() * 0.0
                    old_day_calibration_feature = logits_m.sum() * 0.0
                ce_current = (F.cross_entropy(logits_c, yc, reduction="none") * wc).sum() / (wc.sum() + 1e-8)
                if stage == "head" and float(args.radcil_new_head_boost) != 1.0:
                    # LoRa 的当前轮 batch 本质上都是新伪类；head warmup 阶段
                    # 临时提高当前轮 CE 相对 replay/KD 的权重，验证分类头是否
                    # 因旧类 replay 和 teacher 约束太强而吸收新类不足。该缩放
                    # 只在训练期使用伪标签，不参与聚类或 held-out 选参。
                    new_fraction = (yc >= int(old_out_dim)).float().mean().detach()
                    ce_current = ce_current * (
                        1.0 + (float(args.radcil_new_head_boost) - 1.0) * float(new_fraction.item())
                    )
                ce_memory = F.cross_entropy(logits_m, ym)
                with torch.no_grad():
                    teacher_feat_m, teacher_logits = teacher(xm)
                # masked KD 只约束 teacher 已有输出维度，避免新类伪标签噪声
                # 通过蒸馏目标反向压制当前轮新类别学习。
                if bool(args.radcil_masked_kd):
                    kd_mask = ym < int(old_out_dim)
                else:
                    kd_mask = torch.ones_like(ym, dtype=torch.bool)
                if torch.any(kd_mask):
                    kd = F.kl_div(
                        F.log_softmax(logits_m[kd_mask, :old_out_dim] / float(args.cil_temperature), dim=1),
                        F.softmax(teacher_logits[kd_mask, :old_out_dim] / float(args.cil_temperature), dim=1),
                        reduction="batchmean",
                    ) * (float(args.cil_temperature) ** 2)
                else:
                    kd = logits_m.sum() * 0.0
                if float(args.radcil_feature_distill_weight) > 0:
                    # 特征蒸馏只施加在 replay 样本上，用来约束旧类表征漂移；
                    # 当前轮伪标签样本不参与该项，避免把 teacher 的旧空间强加给新类。
                    feature_distill = F.mse_loss(F.normalize(feat[len(xc):], dim=1), F.normalize(teacher_feat_m, dim=1))
                else:
                    feature_distill = logits_m.sum() * 0.0
                if float(args.radcil_old_replay_supcon_weight) > 0:
                    # Stage67 旧类跨天保持项：只在当前轮开始前已有的 replay 旧类
                    # 样本上做监督对比学习。它不接触当前 discovery 伪标签，也不
                    # 读取 held-out eval 真值，专门验证 Stage66 定位出的 Old Acc
                    # 缺口是否来自旧类类内结构松散。
                    old_replay_mask = ym < int(old_out_dim)
                    if torch.any(old_replay_mask):
                        old_replay_supcon = _supcon_incremental(
                            feat[len(xc):][old_replay_mask],
                            ym[old_replay_mask],
                            args.supcon_temperature,
                        )
                    else:
                        old_replay_supcon = logits_m.sum() * 0.0
                else:
                    old_replay_supcon = logits_m.sum() * 0.0
                if anchor_prototypes is not None:
                    anchor_mask = (ym < int(old_out_dim)) & anchor_valid[ym.clamp(max=int(old_out_dim) - 1)]
                    if torch.any(anchor_mask):
                        student_anchor_features = F.normalize(feat[len(xc):][anchor_mask], dim=1)
                        target_anchor_prototypes = anchor_prototypes[ym[anchor_mask]]
                        prototype_anchor = (
                            1.0 - (student_anchor_features * target_anchor_prototypes).sum(dim=1)
                        ).mean()
                    else:
                        prototype_anchor = logits_m.sum() * 0.0
                else:
                    prototype_anchor = logits_m.sum() * 0.0
                if float(args.radcil_aug_consistency_weight) > 0:
                    # 只对训练 batch 做增强。当前轮伪标签和 replay 标签都
                    # 已经在 discovery/enrollment 或历史训练池内获得，不读取
                    # 任何 held-out evaluation 样本。
                    x_aug = augment_iq_batch(x)
                    feat_aug, _ = student(x_aug)
                    view_consistency = _feature_view_consistency_loss(feat, feat_aug)
                else:
                    view_consistency = logits_m.sum() * 0.0
                if float(args.radcil_domain_alignment_weight) > 0:
                    # 当前轮 discovery 和历史 replay 代表不同采集日。
                    # 只对齐无标签分布统计，不把未知类强行拉到旧类中心。
                    domain_alignment = _feature_domain_alignment_loss(
                        feat[:len(xc)],
                        feat[len(xc):],
                    )
                else:
                    domain_alignment = logits_m.sum() * 0.0
                prototype_assignment = _prototype_assignment_loss(
                    feat[:len(xc)],
                    yc,
                    current_prototypes,
                    current_prototype_valid,
                    temperature=float(args.cil_temperature),
                )
                if float(args.radcil_pseudo_aug_consistency_weight) > 0:
                    confident = wc >= float(args.cil_supcon_threshold)
                    if torch.any(confident):
                        _, augmented_logits = student(augment_iq_batch(xc[confident]))
                        pseudo_aug_consistency = (
                            F.kl_div(
                                F.log_softmax(logits_c[confident] / float(args.cil_temperature), dim=1),
                                F.softmax(augmented_logits / float(args.cil_temperature), dim=1),
                                reduction="batchmean",
                            )
                            * (float(args.cil_temperature) ** 2)
                        )
                    else:
                        pseudo_aug_consistency = logits_c.sum() * 0.0
                else:
                    pseudo_aug_consistency = logits_c.sum() * 0.0
                if float(args.radcil_recording_consistency_weight) > 0 and rc is not None:
                    # 该项只看当前轮 discovery batch 内的 recording_id，不接触
                    # 评估集，也不修改伪标签。它验证 LoRa 多 symbol recording
                    # 结构是否能在 CIL 训练期稳定新类预测。
                    recording_consistency = _recording_logit_consistency_loss(
                        logits_c,
                        yc,
                        rc,
                        temperature=float(args.radcil_recording_consistency_temperature),
                    )
                else:
                    recording_consistency = logits_c.sum() * 0.0
                if float(args.radcil_recording_ce_weight) > 0 and rc is not None:
                    # 与逐 symbol CE 并行的 recording-level CE：同一 recording
                    # 内同伪类 symbol 先聚合 logits，再用同一伪标签监督。该项
                    # 不替换 discovery 标签，也不读取 held-out evaluation。
                    recording_ce = _recording_mean_logit_ce_loss(
                        logits_c,
                        yc,
                        wc,
                        rc,
                    )
                else:
                    recording_ce = logits_c.sum() * 0.0
                high = wc >= float(args.cil_supcon_threshold)
                feat_con = torch.cat([feat[:len(xc)][high], feat[len(xc):]], dim=0)
                y_con = torch.cat([yc[high], ym], dim=0)
                con = _supcon_incremental(feat_con, y_con, args.supcon_temperature)
                # 只使用高置信新类伪标签和 replay 旧类标签做代理度量，
                # 让跨天增量训练直接优化类别角度间隔。
                metric_loss = cosine_proxy_metric_loss(
                    feat_con,
                    y_con,
                    student.classifier.weight,
                    scale=float(args.radcil_metric_scale),
                    margin=float(args.radcil_metric_margin),
                )
                kd_weight = scheduled_kd_weight(epoch_index, epochs)
                loss = (
                    ce_current
                    + float(args.cil_replay_weight) * ce_memory
                    + kd_weight * kd
                    + float(args.radcil_feature_distill_weight) * feature_distill
                    + float(args.radcil_old_replay_supcon_weight) * old_replay_supcon
                    + float(args.radcil_prototype_anchor_weight) * prototype_anchor
                    + float(args.radcil_aug_consistency_weight) * view_consistency
                    + float(args.radcil_domain_alignment_weight) * domain_alignment
                    + float(args.radcil_new_prototype_weight) * prototype_assignment
                    + float(args.radcil_pseudo_aug_consistency_weight) * pseudo_aug_consistency
                    + float(args.radcil_recording_consistency_weight) * recording_consistency
                    + float(args.radcil_recording_ce_weight) * recording_ce
                    + float(args.cil_supcon_weight) * con
                    + float(args.radcil_metric_weight) * metric_loss
                    + float(args.lora_old_day_joint_cil_ce_weight) * old_day_calibration_ce
                    + float(args.lora_old_day_joint_cil_feature_weight) * old_day_calibration_feature
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
                opt.step()
    student.eval()
    return student


def _recalibrate_balanced_classifier_head(
    student,
    current_x,
    current_y,
    current_w,
    memory_x,
    memory_y,
    epochs,
    batch_size,
    device,
    old_out_dim=0,
    new_distill_weight=0.0,
):
    """冻结 backbone，使用等类采样重新校准增量分类头。

    LoRa Chirp 实验暴露出一个具体问题：RADCIL 的 backbone 已经能提供较好的
    跨天特征，但当前轮伪标签样本和旧类 replay 样本经过不同采样路径进入分类头，
    容易形成偏向新类或偏向旧类的决策边界。这里不重新聚类，也不使用 held-out
    标签，而是把当前轮和 replay 合并后按类别等量抽样，在冻结特征上短暂训练
    classifier。这样验证的是“类别先验失衡”是否为主因，而不是继续搜索预测阈值。

    参数：
        student: 已完成当前轮 RADCIL 的模型，返回时仍处于 eval 模式。
        current_x/current_y/current_w: 当前轮 discovery 原始 IQ、伪标签及置信度权重。
        memory_x/memory_y: 当前轮开始前的旧类 replay memory。
        epochs: 重校准轮数；0 表示完全关闭。
        batch_size: 分类头优化 batch size。
        device: 训练设备。
        old_out_dim: 当前轮训练前已有的旧类输出维度，用来区分当前轮新伪类。
        new_distill_weight: 对当前轮新伪类样本保留重校准前 logits 分布的权重。
    """
    epochs = int(epochs)
    if epochs <= 0 or len(current_x) == 0 or len(memory_x) == 0:
        student.eval()
        return student

    current_x = np.asarray(current_x, dtype=np.float32)
    current_y = np.asarray(current_y, dtype=np.int64)
    current_w = np.asarray(current_w, dtype=np.float32)
    memory_x = np.asarray(memory_x, dtype=np.float32)
    memory_y = np.asarray(memory_y, dtype=np.int64)
    if len(current_x) != len(current_y) or len(current_x) != len(current_w):
        raise ValueError("Balanced head recalibration current arrays have inconsistent lengths.")
    if len(memory_x) != len(memory_y):
        raise ValueError("Balanced head recalibration memory arrays have inconsistent lengths.")

    # 每个类别取相同数量，避免 replay 类别数量或当前伪类样本量直接改变 head
    # 的先验。抽样只依赖训练期标签和固定随机种子，不接触任何评估标签。
    rng = np.random.default_rng(17)
    class_to_indices = {}
    all_x = np.concatenate([memory_x, current_x], axis=0)
    all_y = np.concatenate([memory_y, current_y], axis=0)
    all_w = np.concatenate([
        np.ones(len(memory_y), dtype=np.float32),
        np.maximum(current_w, 0.05).astype(np.float32),
    ])
    candidate_indices = {
        int(class_id): np.where(all_y == int(class_id))[0]
        for class_id in sorted(np.unique(all_y).tolist())
    }
    candidate_indices = {
        class_id: indices for class_id, indices in candidate_indices.items() if len(indices) > 0
    }
    if not candidate_indices:
        student.eval()
        return student
    take_per_class = min(32, min(len(indices) for indices in candidate_indices.values()))
    for class_id, class_indices in candidate_indices.items():
        chosen = rng.choice(class_indices, size=take_per_class, replace=False)
        class_to_indices[int(class_id)] = chosen
    selected = np.concatenate(list(class_to_indices.values())).astype(np.int64)
    if len(selected) == 0:
        student.eval()
        return student

    student.eval()
    feature_batches = []
    with torch.no_grad():
        loader = DataLoader(
            TensorDataset(torch.as_tensor(all_x[selected], dtype=torch.float32)),
            batch_size=max(1, int(batch_size)), shuffle=False, num_workers=0,
        )
        for (xb,) in loader:
            features, _ = student(xb.to(device))
            feature_batches.append(features.detach().cpu())
    features = torch.cat(feature_batches, dim=0).to(device)
    labels = torch.as_tensor(all_y[selected], dtype=torch.long, device=device)
    weights = torch.as_tensor(all_w[selected], dtype=torch.float32, device=device)
    selected_current = selected >= len(memory_y)
    selected_new = selected_current & (all_y[selected] >= int(old_out_dim))
    selected_new = torch.as_tensor(selected_new.astype(np.bool_), device=device)
    with torch.no_grad():
        # 重校准前的分类头作为“新类保持”教师。该教师来自同一轮训练结束后的
        # Student，不引入额外模型，也不读取 eval 标签；只在当前轮新伪类样本上
        # 约束 logits 分布，避免类均衡 CE 把新类整体吸回旧类。
        teacher_logits = student.classifier(features).detach()

    # 只打开 classifier 参数；backbone 梯度关闭，保证该实验只检验分类头先验校准。
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    student.classifier.weight.requires_grad_(True)
    student.classifier.bias.requires_grad_(True)
    optimizer = torch.optim.AdamW(student.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(features, labels, weights, teacher_logits, selected_new),
        batch_size=max(1, int(batch_size)), shuffle=True, num_workers=0,
    )
    student.train()
    for _ in range(epochs):
        for xb, yb, wb, tb, nb in loader:
            logits = student.classifier(xb)
            loss = (F.cross_entropy(logits, yb, reduction="none") * wb).sum() / (wb.sum() + 1e-8)
            if float(new_distill_weight) > 0 and torch.any(nb):
                temperature = 2.0
                distill = F.kl_div(
                    F.log_softmax(logits[nb] / temperature, dim=1),
                    F.softmax(tb[nb] / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature * temperature)
                loss = loss + float(new_distill_weight) * distill
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.classifier.parameters(), max_norm=5.0)
            optimizer.step()
    student.eval()
    return student


@torch.no_grad()
def _extract_end_to_end_outputs(model, X, batch_size, device):
    """批量提取网络特征和分类 logits，供纯网络与混合后端共同复用。"""
    loader = DataLoader(TensorDataset(torch.as_tensor(X, dtype=torch.float32)), batch_size=int(batch_size), shuffle=False)
    features, logits = [], []
    model.eval()
    for (xb,) in loader:
        feat_batch, logit_batch = model(xb.to(device))
        features.append(feat_batch.cpu().numpy())
        logits.append(logit_batch.cpu().numpy())
    if not features:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    return (
        np.concatenate(features, axis=0).astype(np.float32),
        np.concatenate(logits, axis=0).astype(np.float32),
    )


def _predict_end_to_end(model, X, batch_size, device, old_class_count=0, old_logit_bias=0.0):
    """使用增量网络分类头预测伪标签类别。

    ``old_logit_bias`` 是一个严格的验证集校准项：它只给当前轮训练前已经
    存在的类别列加同一个 logit 偏置，用来诊断旧类分数是否被新类头系统性
    压低。偏置候选必须来自 Day1/IQ_7 validation，不能读取 held-out eval
    真值；默认值 0 完全保持历史网络预测行为。
    """
    _, logits = _extract_end_to_end_outputs(model, X, batch_size, device)
    if len(logits) and int(old_class_count) > 0 and abs(float(old_logit_bias)) > 0:
        old_class_count = min(int(old_class_count), logits.shape[1])
        logits = logits.copy()
        logits[:, :old_class_count] += float(old_logit_bias)
    return np.argmax(logits, axis=1).astype(np.int64) if len(logits) else np.empty(0, dtype=np.int64)


def _save_end_to_end_eval_dump(
    save_dir,
    stage_name,
    eval_x,
    eval_y,
    pred_raw,
    pseudo_to_true,
    batch_size,
    device,
    model,
    recording_id=None,
):
    """保存端到端评估预测，供离线组合/路由诊断复用。

    该转储默认关闭，只在显式传入 ``--save_end_to_end_eval_dumps`` 时生效。
    文件只记录已经完成训练后的 held-out 评估预测、logits 和当轮报告用
    ``pseudo_to_true`` 映射，不参与训练、发现、阈值校准或模型选择。
    """
    dump_dir = os.path.join(save_dir, "end_to_end_eval_dumps")
    ensure_dir(dump_dir)
    safe_stage = str(stage_name).lower().replace(" ", "_").replace("-", "_")
    _, logits = _extract_end_to_end_outputs(model, eval_x, batch_size, device)
    mapping_keys = np.asarray(sorted(int(k) for k in pseudo_to_true.keys()), dtype=np.int64)
    mapping_values = np.asarray([int(pseudo_to_true[int(k)]) for k in mapping_keys], dtype=np.int64)
    pred_raw = np.asarray(pred_raw, dtype=np.int64)
    pred_mapped = np.asarray([pseudo_to_true.get(int(p), int(p)) for p in pred_raw], dtype=np.int64)
    payload = {
        "y_true": np.asarray(eval_y, dtype=np.int64),
        "pred_raw": pred_raw,
        "pred_mapped": pred_mapped,
        "logits": np.asarray(logits, dtype=np.float32),
        "pseudo_to_true_keys": mapping_keys,
        "pseudo_to_true_values": mapping_values,
    }
    if recording_id is not None:
        payload["recording_id"] = np.asarray(recording_id)
    np.savez_compressed(os.path.join(dump_dir, f"{safe_stage}.npz"), **payload)


def _calibrate_old_logit_bias_on_validation(model, X_val, y_val, batch_size, device, candidates, old_class_count):
    """只用 Day1/IQ_7 验证集选择旧类 logit 偏置。

    IQ_7 只包含初始已知类，因此该校准只衡量“初始旧类是否还能被识别”，
    不评价当前轮新类质量，也不使用任何 held-out eval 样本或真值。返回的
    表格会写入结果目录，方便判断低分是不是旧/新类 logit 尺度失衡造成的。
    """
    rows = []
    best_bias = 0.0
    best_acc = -1.0
    for bias in candidates:
        pred = _predict_end_to_end(
            model, X_val, batch_size, device,
            old_class_count=old_class_count,
            old_logit_bias=float(bias),
        )
        acc = float(np.mean(pred == y_val)) if len(pred) else 0.0
        rows.append({
            "Old Logit Bias": float(bias),
            "Old Class Count": int(old_class_count),
            "IQ_7 Validation Old-Class Acc": acc,
        })
        if acc > best_acc or (np.isclose(acc, best_acc) and abs(float(bias)) < abs(float(best_bias))):
            best_acc = acc
            best_bias = float(bias)
    return best_bias, rows


def _build_aligned_doi_prototype_bank(
    model,
    memory_x,
    memory_y,
    batch_size,
    device,
    previous_bank=None,
    align_lambda=0.30,
):
    """从 replay IQ 记忆构建 DOI-style 对齐原型。

    原型只使用训练集和逐轮伪标签记忆；held-out 评估样本及其真值不会进入
    原型估计。历史类别使用上一轮原型与当前 replay 原型做平滑对齐，降低
    骨干微调导致的旧类中心漂移。
    """
    memory_features, _ = _extract_end_to_end_outputs(model, memory_x, batch_size, device)
    prototypes, prototype_labels = make_prototypes(memory_features, np.asarray(memory_y, dtype=np.int64))

    if previous_bank is not None:
        old_prototypes, old_labels = previous_bank
        current_index = {int(label): index for index, label in enumerate(prototype_labels)}
        for old_index, label in enumerate(old_labels):
            label = int(label)
            if label not in current_index:
                continue
            current_idx = current_index[label]
            prototypes[current_idx] = (
                (1.0 - float(align_lambda)) * old_prototypes[old_index]
                + float(align_lambda) * prototypes[current_idx]
            )

    return l2norm(prototypes).astype(np.float32), np.asarray(prototype_labels, dtype=np.int64)


def _build_replay_prototype_bank(model, memory_x, memory_y, batch_size, device):
    """从 replay IQ 记忆构建 iCaRL-style exemplar 原型库。

    该原型库只使用已训练样本和逐轮伪标签 replay，不做 DOI-style 历史中心
    对齐，也不读取 held-out 评估真值。它用于低置信网络预测的离散回退，
    与 DOI-memory 的连续概率 late-fusion 保持机制区分。
    """
    memory_features, _ = _extract_end_to_end_outputs(model, memory_x, batch_size, device)
    prototypes, prototype_labels = make_prototypes(memory_features, np.asarray(memory_y, dtype=np.int64))
    return l2norm(prototypes).astype(np.float32), np.asarray(prototype_labels, dtype=np.int64)


def _predict_hybrid_radcil_doi(
    model,
    eval_x,
    prototype_bank,
    batch_size,
    device,
    fusion_weight=0.30,
    prototype_temperature=0.10,
    old_class_count=None,
):
    """融合网络分类概率与 DOI-style replay 原型概率。

    `fusion_weight=0` 等价于原网络分类头。原型分数按伪标签类别 ID 对齐
    到分类头输出，网络仍承担真实增量学习，原型记忆只作为后验校正项。
    """
    fusion_weight = float(fusion_weight)
    if not 0.0 <= fusion_weight <= 1.0:
        raise ValueError("--radcil_doi_fusion_weight must be within [0, 1].")
    if float(prototype_temperature) <= 0:
        raise ValueError("--radcil_doi_prototype_temperature must be positive.")

    eval_features, network_logits = _extract_end_to_end_outputs(model, eval_x, batch_size, device)
    if len(network_logits) == 0 or fusion_weight <= 0 or prototype_bank is None:
        return np.argmax(network_logits, axis=1).astype(np.int64) if len(network_logits) else np.empty(0, dtype=np.int64)

    prototypes, prototype_labels = prototype_bank
    prototype_scores = l2norm(eval_features) @ l2norm(prototypes).T
    class_scores = np.full(network_logits.shape, -1e4, dtype=np.float32)
    for prototype_index, label in enumerate(prototype_labels):
        label = int(label)
        if label < 0 or label >= class_scores.shape[1]:
            raise ValueError(f"Prototype label {label} is outside classifier output range.")
        class_scores[:, label] = prototype_scores[:, prototype_index] / float(prototype_temperature)

    network_log_prob = F.log_softmax(torch.from_numpy(network_logits), dim=1).numpy()
    prototype_log_prob = F.log_softmax(torch.from_numpy(class_scores), dim=1).numpy()
    fused_log_prob = network_log_prob.copy()
    # 分组双头融合只校正本轮训练前已经存在的旧类列；当前新注册类继续完全
    # 使用网络分类头。类别边界来自 classifier 扩展前的输出维数，不需要知道
    # 任一样本的真实旧/新身份，因此不会引入 held-out 标签泄漏。
    old_class_count = class_scores.shape[1] if old_class_count is None else int(old_class_count)
    if old_class_count < 0 or old_class_count > class_scores.shape[1]:
        raise ValueError("old_class_count is outside classifier output range.")
    fused_log_prob[:, :old_class_count] = (
        (1.0 - fusion_weight) * network_log_prob[:, :old_class_count]
        + fusion_weight * prototype_log_prob[:, :old_class_count]
    )
    return np.argmax(fused_log_prob, axis=1).astype(np.int64)


def _predict_radcil_icarl_fallback(
    model,
    eval_x,
    prototype_bank,
    batch_size,
    device,
    confidence_threshold=0.60,
    old_class_count=None,
):
    """低置信时回退到 exemplar 原型分类器。

    网络分类头仍给出默认预测；只有当最大 softmax 置信度低于阈值时，才
    使用 replay 原型的最近类结果替换。若提供 `old_class_count`，只允许
    回退到训练前已经存在的旧类列，避免把当前轮新类样本系统性吸回旧类。
    """
    threshold = float(confidence_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--radcil_icarl_fallback_threshold must be within [0, 1].")

    eval_features, network_logits = _extract_end_to_end_outputs(model, eval_x, batch_size, device)
    if len(network_logits) == 0:
        return np.empty(0, dtype=np.int64)
    network_prob = F.softmax(torch.from_numpy(network_logits), dim=1).numpy()
    network_pred = np.argmax(network_prob, axis=1).astype(np.int64)
    if threshold <= 0.0 or prototype_bank is None:
        return network_pred

    prototypes, prototype_labels = prototype_bank
    prototype_pred = predict_proto(eval_features, prototypes, prototype_labels)
    fallback_mask = np.max(network_prob, axis=1) < threshold
    if old_class_count is not None:
        old_class_count = int(old_class_count)
        if old_class_count < 0 or old_class_count > network_logits.shape[1]:
            raise ValueError("old_class_count is outside classifier output range.")
        fallback_mask &= prototype_pred < old_class_count
    pred = network_pred.copy()
    pred[fallback_mask] = prototype_pred[fallback_mask]
    return pred.astype(np.int64)


def _predict_radcil_old_prototype_route(
    model,
    eval_x,
    prototype_bank,
    batch_size,
    device,
    old_mass_threshold=0.55,
    old_class_count=None,
):
    """旧类概率质量门控的原型路由预测。

    Stage 28 显示 DOI-style 原型后端能明显救回 LoRa 旧类，但会把大量
    新类吸回旧类；因此这里不做全局融合，也不在网络低置信时无条件回退。
    只有当网络 softmax 在旧类列上的总概率已经足够高时，才认为该样本
    是“疑似旧类”，并使用 replay 原型只修正旧类内部类别。这样路由信号
    不需要 held-out 真值，也能尽量保留 RADCIL 对当前新类的判断。
    """
    threshold = float(old_mass_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--radcil_old_prototype_route_candidates values must be within [0, 1].")
    if prototype_bank is None:
        return _predict_end_to_end(model, eval_x, batch_size, device)

    eval_features, network_logits = _extract_end_to_end_outputs(model, eval_x, batch_size, device)
    if len(network_logits) == 0:
        return np.empty(0, dtype=np.int64)
    old_class_count = network_logits.shape[1] if old_class_count is None else int(old_class_count)
    if old_class_count <= 0 or old_class_count > network_logits.shape[1]:
        raise ValueError("old_class_count is outside classifier output range.")

    network_prob = F.softmax(torch.from_numpy(network_logits), dim=1).numpy()
    pred = np.argmax(network_prob, axis=1).astype(np.int64)
    old_mass = np.sum(network_prob[:, :old_class_count], axis=1)

    prototypes, prototype_labels = prototype_bank
    old_mask = np.asarray(prototype_labels, dtype=np.int64) < old_class_count
    if not np.any(old_mask):
        return pred
    old_prototypes = np.asarray(prototypes, dtype=np.float32)[old_mask]
    old_labels = np.asarray(prototype_labels, dtype=np.int64)[old_mask]
    old_scores = l2norm(eval_features) @ l2norm(old_prototypes).T
    old_pred = old_labels[np.argmax(old_scores, axis=1)]

    route_mask = old_mass >= threshold
    pred[route_mask] = old_pred[route_mask]
    return pred.astype(np.int64)


def _calibrate_grouped_doi_weight_on_validation(
    model, validation_x, validation_y, prototype_bank, batch_size, device,
    candidates, prototype_temperature, old_class_count,
):
    """仅使用 Day1 validation 已知类选择旧类原型融合权重。

    IQ_7 不包含增量新类，因此它只负责约束旧类保持；新类列无论选择哪个
    候选都保留网络头分数。并列时选择较小权重，减少不必要的原型偏置。
    """
    candidate_weights = sorted({float(value) for value in candidates})
    if not candidate_weights:
        raise ValueError("Grouped DOI fusion requires at least one calibration candidate.")
    if any(weight < 0.0 or weight > 1.0 for weight in candidate_weights):
        raise ValueError("Grouped DOI fusion candidates must be within [0, 1].")

    rows = []
    for weight in candidate_weights:
        prediction = _predict_hybrid_radcil_doi(
            model, validation_x, prototype_bank, batch_size, device, weight,
            prototype_temperature, old_class_count=old_class_count,
        )
        rows.append({
            "Fusion Weight": weight,
            "Validation Accuracy": float(np.mean(prediction == validation_y)),
        })
    best = sorted(rows, key=lambda row: (-row["Validation Accuracy"], row["Fusion Weight"]))[0]
    return float(best["Fusion Weight"]), rows


def _calibrate_old_prototype_route_on_validation(
    model, validation_x, validation_y, prototype_bank, batch_size, device,
    candidates, old_class_count,
):
    """仅用 Day1/IQ_7 旧类验证集选择旧类路由阈值。

    验证集没有增量新类，不能用来估计新类误吸风险；因此并列时选择更高
    阈值，使路由更保守。最终 held-out 新类只用于事后报告，不参与选参。
    """
    candidate_thresholds = sorted({float(value) for value in candidates})
    if not candidate_thresholds:
        raise ValueError("Old prototype route requires at least one threshold candidate.")
    if any(value < 0.0 or value > 1.0 for value in candidate_thresholds):
        raise ValueError("Old prototype route candidates must be within [0, 1].")

    rows = []
    for threshold in candidate_thresholds:
        prediction = _predict_radcil_old_prototype_route(
            model, validation_x, prototype_bank, batch_size, device,
            old_mass_threshold=threshold, old_class_count=old_class_count,
        )
        rows.append({
            "Old Mass Threshold": threshold,
            "Validation Accuracy": float(np.mean(prediction == validation_y)),
        })
    best = sorted(rows, key=lambda row: (-row["Validation Accuracy"], -row["Old Mass Threshold"]))[0]
    return float(best["Old Mass Threshold"]), rows


def _calibrate_icarl_fallback_threshold_on_validation(
    model, validation_x, validation_y, prototype_bank, batch_size, device,
    candidates, old_class_count,
):
    """仅用 Day1 validation 旧类选择低置信 fallback 阈值。

    WiSig 使用 Day1 development 内部切出的 10% 验证样本，LoRa 使用固定
    IQ_7 验证 transmission；二者都只包含初始已知类。因此这个校准只能
    用于约束旧类保持，不能评价增量新类。并列时选择更低阈值，使
    fallback 尽量少触发，避免把该机制退化成纯原型分类器。
    """
    candidate_thresholds = sorted({float(value) for value in candidates})
    if not candidate_thresholds:
        raise ValueError("iCaRL fallback requires at least one calibration candidate.")
    if any(value < 0.0 or value > 1.0 for value in candidate_thresholds):
        raise ValueError("iCaRL fallback candidates must be within [0, 1].")

    rows = []
    for threshold in candidate_thresholds:
        prediction = _predict_radcil_icarl_fallback(
            model, validation_x, prototype_bank, batch_size, device,
            confidence_threshold=threshold, old_class_count=old_class_count,
        )
        rows.append({
            "Fallback Threshold": threshold,
            "Validation Accuracy": float(np.mean(prediction == validation_y)),
        })
    best = sorted(rows, key=lambda row: (-row["Validation Accuracy"], row["Fallback Threshold"]))[0]
    return float(best["Fallback Threshold"]), rows


def _prototype_predict_from_memory(memory_X, memory_y, eval_X, old_class_bonus=0.0, old_class_count=10):
    if memory_X is None or len(memory_X) == 0:
        return np.zeros(len(eval_X), dtype=np.int64)
    protos, proto_labels = make_prototypes(memory_X, memory_y)
    return predict_proto(eval_X, protos, proto_labels, old_class_count=old_class_count, old_class_bonus=old_class_bonus)


def _smooth_prototypes_by_graph(prototypes, smooth_lambda=0.2):
    """TPCIL-style topology smoothing over class prototypes."""
    P = l2norm(prototypes)
    if len(P) <= 1 or smooth_lambda <= 0:
        return P
    S = np.clip((P @ P.T + 1.0) / 2.0, 0.0, 1.0)
    np.fill_diagonal(S, 0.0)
    row_sum = S.sum(axis=1, keepdims=True) + 1e-8
    A = S / row_sum
    P_smooth = (1.0 - float(smooth_lambda)) * P + float(smooth_lambda) * (A @ P)
    return l2norm(P_smooth)


def _tpcil_style_predict(memory_X, memory_y, eval_X, smooth_lambda=0.2):
    if memory_X is None or len(memory_X) == 0:
        return np.zeros(len(eval_X), dtype=np.int64)
    protos, proto_labels = make_prototypes(memory_X, memory_y)
    protos = _smooth_prototypes_by_graph(protos, smooth_lambda=smooth_lambda)
    sim = l2norm(eval_X) @ l2norm(protos).T
    return proto_labels[np.argmax(sim, axis=1)]




def _doi_style_predict(memory_X, memory_y, eval_X, old_proto_memory=None, align_lambda=0.30):
    """
    DOI-style SEI incremental baseline.

    This is an adapted SEI-oriented prototype baseline rather than an exact
    reproduction of DOI. It follows the main DOI ideas:
        1) prototype memory preservation
        2) old/new prototype alignment
        3) incremental prototype update

    All evaluations use the same WiSig Cross-Day protocol and feature space.
    """
    if memory_X is None or len(memory_X) == 0:
        return np.zeros(len(eval_X), dtype=np.int64)

    protos, proto_labels = make_prototypes(memory_X, memory_y)

    # Prototype alignment: preserve historical knowledge while adapting to new data.
    if old_proto_memory is not None:
        old_protos, old_labels = old_proto_memory
        label_to_idx = {int(c): i for i, c in enumerate(proto_labels)}
        for i, c in enumerate(old_labels):
            c = int(c)
            if c in label_to_idx:
                j = label_to_idx[c]
                protos[j] = (
                    (1.0 - float(align_lambda)) * old_protos[i]
                    + float(align_lambda) * protos[j]
                )
        protos = l2norm(protos)

    sim = l2norm(eval_X) @ l2norm(protos).T
    return proto_labels[np.argmax(sim, axis=1)]


def _update_doi_prototype_memory(memory_X, memory_y, new_X, new_y):
    """DOI-style memory update: concatenate old/new samples."""
    if memory_X is None or len(memory_X) == 0:
        return new_X.copy(), new_y.copy()
    if new_X is None or len(new_X) == 0:
        return memory_X, memory_y
    return (
        np.concatenate([memory_X, new_X], axis=0),
        np.concatenate([memory_y, new_y], axis=0),
    )


def build_comparison_summary(baseline_df, incremental_df, proposed_source_method="MV-ACC", proposed_name="Ours (MV-ACC)"):
    rows = []
    if baseline_df is not None and not baseline_df.empty:
        df = baseline_df.copy()
        df = df[df["Stage"].astype(str).str.startswith("After R")]
        for method in df["Method"].unique().tolist():
            sub = df[df["Method"] == method]
            item = {"Method": method}
            for r in [1, 2, 3]:
                rr = sub[sub["Stage"] == f"After R{r}"]
                if len(rr) > 0:
                    item[f"R{r} Acc"] = float(rr.iloc[0]["Overall Acc"])
                    item[f"R{r} Forgetting"] = float(rr.iloc[0]["Forgetting Rate"])
                else:
                    item[f"R{r} Acc"] = np.nan
                    item[f"R{r} Forgetting"] = np.nan
            item["Avg Acc"] = float(np.nanmean([item.get("R1 Acc"), item.get("R2 Acc"), item.get("R3 Acc")]))
            item["Avg Forgetting"] = float(np.nanmean([item.get("R1 Forgetting"), item.get("R2 Forgetting"), item.get("R3 Forgetting")]))
            rows.append(item)

    # Add the proposed method from MV-ACC rows in the main incremental table.
    if incremental_df is not None and not incremental_df.empty:
        ours = incremental_df[(incremental_df["Method"] == proposed_source_method) & (incremental_df["Stage"].astype(str).str.startswith("After R"))]
        if len(ours) > 0:
            item = {"Method": proposed_name}
            for r in [1, 2, 3]:
                rr = ours[ours["Stage"] == f"After R{r}"]
                if len(rr) > 0:
                    item[f"R{r} Acc"] = float(rr.iloc[0]["Overall Acc"])
                    item[f"R{r} Forgetting"] = float(rr.iloc[0]["Forgetting Rate"])
                else:
                    item[f"R{r} Acc"] = np.nan
                    item[f"R{r} Forgetting"] = np.nan
            item["Avg Acc"] = float(np.nanmean([item.get("R1 Acc"), item.get("R2 Acc"), item.get("R3 Acc")]))
            item["Avg Forgetting"] = float(np.nanmean([item.get("R1 Forgetting"), item.get("R2 Forgetting"), item.get("R3 Forgetting")]))
            rows.append(item)

    return pd.DataFrame(rows)




def build_full_end_to_end_system_comparison(
    end_to_end_baseline_df,
    incremental_df,
    proposed_source_method="MV-ACC-CIL",
    proposed_display_name="Ours (MV-ACC)",
    proposed_incremental_module="Network replay + distillation",
):
    """
    Build the paper-facing end-to-end system table.

    This table combines each discovery front-end with its downstream incremental
    learner and therefore measures discovery quality and incremental retention
    jointly. Classical CIL methods do not contain an unknown-discovery module,
    so they are coupled with the neutral Deep-HDBSCAN front-end.

    Important: TPCIL-style and DOI-style remain adapted back-end implementations;
    they are not claimed as exact reproductions of the original complete systems.
    """
    rows = []

    def add_from_incremental(source_method, display_name, discovery, learner):
        sub = incremental_df[(incremental_df["Method"] == source_method) &
                             (incremental_df["Stage"].astype(str).str.startswith("After R"))]
        if sub.empty:
            return
        item = {
            "Method": display_name,
            "Discovery Module": discovery,
            "Incremental Module": learner,
        }
        for r in [1, 2, 3]:
            rr = sub[sub["Stage"] == f"After R{r}"]
            item[f"R{r} Acc"] = float(rr.iloc[0]["Overall Acc"]) if len(rr) else np.nan
            item[f"R{r} Forgetting"] = float(rr.iloc[0]["Forgetting Rate"]) if len(rr) else np.nan
        item["Avg Acc"] = float(np.nanmean([item[f"R{r} Acc"] for r in [1,2,3]]))
        item["Avg Forgetting"] = float(np.nanmean([item[f"R{r} Forgetting"] for r in [1,2,3]]))
        rows.append(item)

    def add_from_baseline(source_method, display_name, discovery, learner):
        if end_to_end_baseline_df is None or end_to_end_baseline_df.empty:
            return
        sub = end_to_end_baseline_df[(end_to_end_baseline_df["Method"] == source_method) &
                                     (end_to_end_baseline_df["Stage"].astype(str).str.startswith("After R"))]
        if sub.empty:
            return
        item = {
            "Method": display_name,
            "Discovery Module": discovery,
            "Incremental Module": learner,
        }
        for r in [1, 2, 3]:
            rr = sub[sub["Stage"] == f"After R{r}"]
            item[f"R{r} Acc"] = float(rr.iloc[0]["Overall Acc"]) if len(rr) else np.nan
            item[f"R{r} Forgetting"] = float(rr.iloc[0]["Forgetting Rate"]) if len(rr) else np.nan
        item["Avg Acc"] = float(np.nanmean([item[f"R{r} Acc"] for r in [1,2,3]]))
        item["Avg Forgetting"] = float(np.nanmean([item[f"R{r} Forgetting"] for r in [1,2,3]]))
        rows.append(item)

    # Single-view end-to-end systems using their own clustering outputs.
    add_from_incremental(
        "Deep only", "Deep-only pipeline",
        "SupCon deep embedding + HDBSCAN",
        "Prototype enrollment + calibration",
    )
    add_from_incremental(
        "RF only", "RF-only pipeline",
        "Handcrafted RF features + HDBSCAN",
        "Prototype enrollment + calibration",
    )

    # Conventional CIL learners connected to a neutral discovery front-end.
    add_from_baseline("Deep-HDBSCAN + Ft-CNN", "Deep-HDBSCAN + Ft-CNN",
                      "Deep embedding + HDBSCAN", "Fine-tuning")
    add_from_baseline("Deep-HDBSCAN + LwF", "Deep-HDBSCAN + LwF",
                      "Deep embedding + HDBSCAN", "Knowledge distillation")
    add_from_baseline("Deep-HDBSCAN + iCaRL", "Deep-HDBSCAN + iCaRL",
                      "Deep embedding + HDBSCAN", "Exemplar replay + prototype classifier")
    add_from_baseline("Deep-HDBSCAN + EEIL", "Deep-HDBSCAN + EEIL",
                      "Deep embedding + HDBSCAN", "Replay + distillation")
    add_from_baseline("Deep-HDBSCAN + TPCIL-style", "Deep-HDBSCAN + TPCIL-style",
                      "Deep embedding + HDBSCAN", "Topology-aware prototype smoothing")
    add_from_baseline("Deep-HDBSCAN + DOI-style", "Deep-HDBSCAN + DOI-style",
                      "Deep embedding + HDBSCAN", "DOI-inspired prototype preservation/alignment")

    # Proposed complete pipeline.
    add_from_incremental(
        proposed_source_method,
        proposed_display_name,
        "SupCon deep/RF graphs + adaptive fusion + HDBSCAN",
        proposed_incremental_module,
    )

    columns = [
        "Method", "Discovery Module", "Incremental Module",
        "R1 Acc", "R1 Forgetting", "R2 Acc", "R2 Forgetting",
        "R3 Acc", "R3 Forgetting", "Avg Acc", "Avg Forgetting",
    ]
    return pd.DataFrame(rows, columns=columns)

def run_incremental_learning_baselines(discovery_round_infos, proto_bank, y_train, round_data, eval_data, eval_y_dict, args, device, method_prefix="", comparison_label="Shared discovery", feat_type="hybrid"):
    """
    Run representative class-incremental learners using a supplied discovery front-end.

    Parameters
    ----------
    discovery_round_infos:
        Per-round labels and enrolled cluster IDs produced by the selected unknown-discovery
        front-end. Passing Graph-fusion results gives the shared-discovery control experiment;
        passing Deep-only HDBSCAN results gives the end-to-end conventional-CIL pipelines.
    method_prefix:
        Prefix added to method names so the table states the complete pipeline explicitly,
        e.g. ``Deep-HDBSCAN + iCaRL``.
    comparison_label:
        Human-readable label printed in the console.
    feat_type:
        Feature bank used by the incremental learner. The shared-discovery control keeps
        the original hybrid feature setting, while the neutral Deep-HDBSCAN end-to-end
        pipelines use deep features only.
    """
    print(f"\n========== Representative Class-Incremental Baselines: {comparison_label} ==========")

    def mname(base):
        return f"{method_prefix}{base}" if method_prefix else base
    if feat_type not in {"deep", "rf", "hybrid"}:
        raise ValueError(f"Unsupported feat_type={feat_type}")
    train_X = proto_bank["train"][feat_type]
    eval_initial_X = proto_bank["eval_initial"][feat_type]
    in_dim = int(train_X.shape[1])
    init_class_ids = list(range(args.initial_known_classes))

    # Prepare pseudo-labeled data chunks from the supplied discovery front-end.
    pseudo_chunks = []
    pseudo_to_true_global = {}
    next_label = int(args.initial_known_classes)
    for i, info in enumerate(discovery_round_infos, start=1):
        Xp, yp, cluster_pseudo_pairs, next_label = _make_pseudo_labeled_round(
            proto_bank[f"r{i}"][feat_type],
            info["labels"],
            info["enrolled_ids"],
            next_label,
        )
        mapping = build_posthoc_pseudo_to_true(
            round_data[i - 1]["y"], info["labels"], cluster_pseudo_pairs
        )
        pseudo_to_true_global.update(mapping)
        pseudo_chunks.append({
            "X": Xp,
            "y": yp,
            "mapping": mapping,
            "discovered": info["discovered_clusters"],
            "enrolled": info["enrolled_clusters"],
        })

    rows = []

    # Shared initial linear classifier for Ft-CNN/LwF/EEIL.
    base_linear = _make_linear(in_dim, len(init_class_ids), device)
    base_linear = _train_linear_classifier(base_linear, train_X, y_train, init_class_ids, args, device)

    def eval_initial_linear(method, model, class_ids):
        pred = _predict_linear_classifier(model, eval_initial_X, class_ids, device)
        row = _evaluate_raw_predictions(
            method=method,
            stage="Initial",
            eval_day=eval_data["eval_initial"]["day"],
            seen_classes=args.initial_known_classes,
            eval_X_name="day1_eval_initial_30pct",
            y_eval=eval_y_dict["eval_initial"],
            pred_raw=pred,
            pseudo_to_true={},
            true_new="-",
            discovered="-",
            enrolled="-",
            initial_known=args.initial_known_classes,
            round_size=args.round_size,
            initial_reference_acc=None,
        )
        return row, float(row["Initial Known Acc"])

    # Ft-CNN: fine-tune on current pseudo-labeled new data only, no replay, no distillation.
    ft_model = copy.deepcopy(base_linear)
    ft_classes = init_class_ids.copy()
    ft_pseudo_to_true = {}
    init_row, ft_ref = eval_initial_linear(mname("Ft-CNN"), ft_model, ft_classes)
    rows.append(init_row)

    # LwF: fine-tune on current pseudo-labeled data with KD from previous model.
    lwf_model = copy.deepcopy(base_linear)
    lwf_classes = init_class_ids.copy()
    lwf_pseudo_to_true = {}
    init_row, lwf_ref = eval_initial_linear(mname("LwF"), lwf_model, lwf_classes)
    rows.append(init_row)

    # EEIL: exemplar replay + distillation.
    eeil_model = copy.deepcopy(base_linear)
    eeil_classes = init_class_ids.copy()
    eeil_memory_X, eeil_memory_y = _select_exemplars(train_X, y_train, per_class=args.memory_per_class)
    eeil_pseudo_to_true = {}
    init_row, eeil_ref = eval_initial_linear(mname("EEIL"), eeil_model, eeil_classes)
    rows.append(init_row)

    # iCaRL and TPCIL-style use exemplar/prototype classifiers in the frozen feature space.
    icarl_memory_X, icarl_memory_y = _select_exemplars(train_X, y_train, per_class=args.memory_per_class)
    tpcil_memory_X, tpcil_memory_y = icarl_memory_X.copy(), icarl_memory_y.copy()
    icarl_pseudo_to_true = {}
    tpcil_pseudo_to_true = {}
    doi_pseudo_to_true = {}
    doi_memory_X, doi_memory_y = icarl_memory_X.copy(), icarl_memory_y.copy()
    doi_old_proto = make_prototypes(doi_memory_X, doi_memory_y)

    pred_icarl_init = _prototype_predict_from_memory(icarl_memory_X, icarl_memory_y, eval_initial_X, old_class_bonus=0.0, old_class_count=args.initial_known_classes)
    row_icarl_init = _evaluate_raw_predictions(mname("iCaRL"), "Initial", eval_data["eval_initial"]["day"], args.initial_known_classes, "day1_eval_initial_30pct", eval_y_dict["eval_initial"], pred_icarl_init, {}, "-", "-", "-", args.initial_known_classes, args.round_size, None)
    rows.append(row_icarl_init)
    icarl_ref = float(row_icarl_init["Initial Known Acc"])

    pred_tpcil_init = _tpcil_style_predict(tpcil_memory_X, tpcil_memory_y, eval_initial_X, smooth_lambda=args.tpcil_smoothing)
    row_tpcil_init = _evaluate_raw_predictions(mname("TPCIL-style"), "Initial", eval_data["eval_initial"]["day"], args.initial_known_classes, "day1_eval_initial_30pct", eval_y_dict["eval_initial"], pred_tpcil_init, {}, "-", "-", "-", args.initial_known_classes, args.round_size, None)
    rows.append(row_tpcil_init)
    tpcil_ref = float(row_tpcil_init["Initial Known Acc"])

    pred_doi_init = _doi_style_predict(doi_memory_X, doi_memory_y, eval_initial_X, old_proto_memory=None)
    row_doi_init = _evaluate_raw_predictions(mname("DOI-style"), "Initial", eval_data["eval_initial"]["day"], args.initial_known_classes, "day1_eval_initial_30pct", eval_y_dict["eval_initial"], pred_doi_init, {}, "-", "-", "-", args.initial_known_classes, args.round_size, None)
    rows.append(row_doi_init)
    doi_ref = float(row_doi_init["Initial Known Acc"])

    for i, chunk in enumerate(pseudo_chunks, start=1):
        eval_key = f"eval_r{i}"
        eval_X = proto_bank[eval_key][feat_type]
        seen_classes = args.initial_known_classes + i * args.round_size
        stage = f"After R{i}"
        eval_name = f"{eval_key}_30pct_{seen_classes}_seen"

        # Ft-CNN
        new_ids = sorted(np.unique(chunk["y"]).astype(int).tolist())
        old_model = copy.deepcopy(ft_model)
        old_dim = len(ft_classes)
        ft_model, ft_classes = _expand_linear(ft_model, in_dim, ft_classes, new_ids, device)
        ft_model = _train_linear_classifier(ft_model, chunk["X"], chunk["y"], ft_classes, args, device, teacher_model=None, old_output_dim=0)
        ft_pseudo_to_true.update(chunk["mapping"])
        pred = _predict_linear_classifier(ft_model, eval_X, ft_classes, device)
        rows.append(_evaluate_raw_predictions(mname("Ft-CNN"), stage, eval_data[eval_key]["day"], seen_classes, eval_name, eval_y_dict[eval_key], pred, ft_pseudo_to_true, args.round_size, chunk["discovered"], chunk["enrolled"], args.initial_known_classes, args.round_size, ft_ref))

        # LwF
        new_ids = sorted(np.unique(chunk["y"]).astype(int).tolist())
        teacher = copy.deepcopy(lwf_model)
        old_dim = len(lwf_classes)
        lwf_model, lwf_classes = _expand_linear(lwf_model, in_dim, lwf_classes, new_ids, device)
        lwf_model = _train_linear_classifier(lwf_model, chunk["X"], chunk["y"], lwf_classes, args, device, teacher_model=teacher, old_output_dim=old_dim)
        lwf_pseudo_to_true.update(chunk["mapping"])
        pred = _predict_linear_classifier(lwf_model, eval_X, lwf_classes, device)
        rows.append(_evaluate_raw_predictions(mname("LwF"), stage, eval_data[eval_key]["day"], seen_classes, eval_name, eval_y_dict[eval_key], pred, lwf_pseudo_to_true, args.round_size, chunk["discovered"], chunk["enrolled"], args.initial_known_classes, args.round_size, lwf_ref))

        # iCaRL
        icarl_memory_X, icarl_memory_y = _update_exemplar_memory(icarl_memory_X, icarl_memory_y, chunk["X"], chunk["y"], per_class=args.memory_per_class)
        icarl_pseudo_to_true.update(chunk["mapping"])
        pred = _prototype_predict_from_memory(icarl_memory_X, icarl_memory_y, eval_X, old_class_bonus=0.0, old_class_count=args.initial_known_classes)
        rows.append(_evaluate_raw_predictions(mname("iCaRL"), stage, eval_data[eval_key]["day"], seen_classes, eval_name, eval_y_dict[eval_key], pred, icarl_pseudo_to_true, args.round_size, chunk["discovered"], chunk["enrolled"], args.initial_known_classes, args.round_size, icarl_ref))

        # EEIL
        eeil_memory_X, eeil_memory_y = _update_exemplar_memory(eeil_memory_X, eeil_memory_y, chunk["X"], chunk["y"], per_class=args.memory_per_class)
        replay_X = np.concatenate([chunk["X"], eeil_memory_X], axis=0)
        replay_y = np.concatenate([chunk["y"], eeil_memory_y], axis=0)
        new_ids = sorted(np.unique(chunk["y"]).astype(int).tolist())
        teacher = copy.deepcopy(eeil_model)
        old_dim = len(eeil_classes)
        eeil_model, eeil_classes = _expand_linear(eeil_model, in_dim, eeil_classes, new_ids, device)
        eeil_model = _train_linear_classifier(eeil_model, replay_X, replay_y, eeil_classes, args, device, teacher_model=teacher, old_output_dim=old_dim)
        eeil_pseudo_to_true.update(chunk["mapping"])
        pred = _predict_linear_classifier(eeil_model, eval_X, eeil_classes, device)
        rows.append(_evaluate_raw_predictions(mname("EEIL"), stage, eval_data[eval_key]["day"], seen_classes, eval_name, eval_y_dict[eval_key], pred, eeil_pseudo_to_true, args.round_size, chunk["discovered"], chunk["enrolled"], args.initial_known_classes, args.round_size, eeil_ref))

        # TPCIL-style
        tpcil_memory_X, tpcil_memory_y = _update_exemplar_memory(tpcil_memory_X, tpcil_memory_y, chunk["X"], chunk["y"], per_class=args.memory_per_class)
        tpcil_pseudo_to_true.update(chunk["mapping"])
        pred = _tpcil_style_predict(tpcil_memory_X, tpcil_memory_y, eval_X, smooth_lambda=args.tpcil_smoothing)
        rows.append(_evaluate_raw_predictions(mname("TPCIL-style"), stage, eval_data[eval_key]["day"], seen_classes, eval_name, eval_y_dict[eval_key], pred, tpcil_pseudo_to_true, args.round_size, chunk["discovered"], chunk["enrolled"], args.initial_known_classes, args.round_size, tpcil_ref))

        # DOI-style: SEI-specific prototype preservation and alignment baseline.
        doi_memory_X, doi_memory_y = _update_doi_prototype_memory(doi_memory_X, doi_memory_y, chunk["X"], chunk["y"])
        doi_pseudo_to_true.update(chunk["mapping"])
        pred = _doi_style_predict(doi_memory_X, doi_memory_y, eval_X, old_proto_memory=doi_old_proto, align_lambda=0.30)
        rows.append(_evaluate_raw_predictions(mname("DOI-style"), stage, eval_data[eval_key]["day"], seen_classes, eval_name, eval_y_dict[eval_key], pred, doi_pseudo_to_true, args.round_size, chunk["discovered"], chunk["enrolled"], args.initial_known_classes, args.round_size, doi_ref))

    return pd.DataFrame(rows)


def plot_core_method_visualization_panels(args):
    """Create paper-friendly side-by-side panels for the three basic discovery views."""
    if not args.enable_visualization:
        return
    method_dirs = [
        ("Deep only", "deep_only"),
        ("RF only", "rf_only"),
        ("Graph fusion (raw)", "graph_fusion_raw"),
    ]
    out_dir = os.path.join(args.save_dir, "visualizations", "core_comparison")
    ensure_dir(out_dir)

    for r in ["r1", "r2", "r3"]:
        csv_paths = []
        ok = True
        for _, d in method_dirs:
            path = os.path.join(args.save_dir, "visualizations", d, r, "embedding_2d.csv")
            csv_paths.append(path)
            if not os.path.exists(path):
                ok = False
        if not ok:
            continue

        for label_col, suffix, title_suffix in [
            ("true_label", "true_labels", "True Tx labels"),
            ("cluster_label", "clusters", "Discovered clusters"),
        ]:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            for ax, (method_name, _), path in zip(axes, method_dirs, csv_paths):
                df = pd.read_csv(path)
                labels = df[label_col].to_numpy()
                xs = df["x"].to_numpy()
                ys = df["y"].to_numpy()
                unique = sorted(np.unique(labels).tolist())
                non_noise = [u for u in unique if int(u) != -1]
                cmap = plt.get_cmap("tab20", max(len(non_noise), 1))
                if -1 in unique:
                    idx = labels == -1
                    ax.scatter(xs[idx], ys[idx], s=6, c="lightgray", alpha=0.5, edgecolors="none")
                for j, lab in enumerate(non_noise):
                    idx = labels == lab
                    ax.scatter(xs[idx], ys[idx], s=6, color=cmap(j), alpha=0.85, edgecolors="none")
                ax.set_title(method_name)
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"{r.upper()} core method comparison: {title_suffix}")
            plt.tight_layout()
            save_path = os.path.join(out_dir, f"{r}_{suffix}_deep_rf_graph.png")
            plt.savefig(save_path, dpi=300)
            plt.close(fig)

# ============================================================
# Visualization utilities
# ============================================================
# Visualization utilities
# ============================================================

def safe_name(name):
    name = str(name).lower()
    for a, b in [("+", "plus"), (" ", "_"), ("/", "_"), ("\\", "_"), ("-", "_"), ("(", ""), (")", "")]:
        name = name.replace(a, b)
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


def project_2d(features, method="umap", seed=7):
    """
    Visualization projection.

    Default: UMAP with fixed parameters for fair comparison among
    Deep-only, RF-only and Graph fusion.
    """
    features = np.asarray(features, dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    method = method.lower()

    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(features).astype(np.float32)

    if method == "umap":
        if umap is None:
            raise ImportError("Please install UMAP first: pip install umap-learn")

        # PCA pre-reduction before UMAP:
        # 1) remove noisy high-dimensional directions
        # 2) make local manifold structure easier for UMAP to visualize
        # This transformation is applied identically to all compared methods.
        pca_dim = min(50, features.shape[0] - 1, features.shape[1])
        if pca_dim >= 2:
            features = PCA(
                n_components=pca_dim,
                random_state=seed,
            ).fit_transform(features).astype(np.float32)

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.05,
            metric="cosine",
            random_state=seed,
        )
        return reducer.fit_transform(features).astype(np.float32)

    if method == "tsne":
        n = features.shape[0]
        perplexity = min(30, max(5, n // 200))
        perplexity = min(perplexity, n - 1)
        return TSNE(
            n_components=2,
            random_state=seed,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        ).fit_transform(features).astype(np.float32)

    raise ValueError(f"Unsupported visualization method: {method}")


def plot_labeled_embedding(points, labels, title, save_path, noise_label=-1):
    ensure_dir(os.path.dirname(save_path))
    points = np.asarray(points)
    labels = np.asarray(labels)
    unique = sorted(np.unique(labels).tolist())
    non_noise = [x for x in unique if x != noise_label]
    cmap = plt.get_cmap("tab20", max(len(non_noise), 1))

    plt.figure(figsize=(8, 6))

    if noise_label in unique:
        idx = labels == noise_label
        plt.scatter(points[idx, 0], points[idx, 1], s=8, c="lightgray", alpha=0.6, label="noise", edgecolors="none")

    for i, lab in enumerate(non_noise):
        idx = labels == lab
        plt.scatter(points[idx, 0], points[idx, 1], s=8, color=cmap(i), alpha=0.85, label=str(lab), edgecolors="none")

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    if len(unique) <= 20:
        plt.legend(fontsize=7, markerscale=1.8, loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_reliability_embedding(points, cluster_labels, accepted_ids, title, save_path):
    ensure_dir(os.path.dirname(save_path))
    points = np.asarray(points)
    cluster_labels = np.asarray(cluster_labels)
    accepted_ids = set(int(x) for x in accepted_ids)

    status = np.full(len(cluster_labels), "rejected", dtype=object)
    status[cluster_labels == -1] = "noise"
    for cid in accepted_ids:
        status[cluster_labels == cid] = "accepted"

    plt.figure(figsize=(8, 6))
    for name, color, alpha in [("noise", "lightgray", 0.6), ("rejected", "orange", 0.75), ("accepted", "green", 0.85)]:
        idx = status == name
        if np.sum(idx) > 0:
            plt.scatter(points[idx, 0], points[idx, 1], s=8, c=color, alpha=alpha, label=name, edgecolors="none")

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(fontsize=8, markerscale=1.8, loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_discovery_visualizations(method, round_name, day_name, features, y_true, cluster_labels, accepted_ids, args, full_method=False):
    if not args.enable_visualization:
        return

    out_dir = os.path.join(args.save_dir, "visualizations", safe_name(method), round_name.lower())
    ensure_dir(out_dir)

    projection_methods = ["umap", "tsne"] if args.visualization_method == "both" else [args.visualization_method]
    for projection_method in projection_methods:
        print(f"[Visualize] {method} {round_name}: {projection_method}")
        points = project_2d(features, method=projection_method, seed=args.seed)
        suffix = f"_{projection_method}" if args.visualization_method == "both" else ""
        plot_labeled_embedding(points, y_true,
            title=f"{method} {round_name} {day_name}: true Tx labels ({projection_method.upper()})",
            save_path=os.path.join(out_dir, f"true_labels{suffix}.png"), noise_label=-999999)
        plot_labeled_embedding(points, cluster_labels,
            title=f"{method} {round_name} {day_name}: HDBSCAN clusters ({projection_method.upper()})",
            save_path=os.path.join(out_dir, f"clusters{suffix}.png"), noise_label=-1)
        if full_method:
            plot_reliability_embedding(points, cluster_labels, accepted_ids,
                title=f"{method} {round_name} {day_name}: reliability status ({projection_method.upper()})",
                save_path=os.path.join(out_dir, f"reliability_status{suffix}.png"))
        embedding = pd.DataFrame({"x": points[:, 0], "y": points[:, 1],
                                  "true_label": y_true, "cluster_label": cluster_labels})
        embedding.to_csv(os.path.join(out_dir, f"embedding_2d{suffix}.csv"), index=False, encoding="utf-8-sig")

# ============================================================
# Main experiment
# ============================================================

def calibrate_mvacc_on_known_day1(X_cal, y_cal, Z_cal, args):
    """Freeze density and consolidation settings using Day1 validation only."""
    rows = []
    for ratio in [float(x) for x in args.mvacc_calibration_ratios.split(",") if x.strip()]:
        for threshold in [float(x) for x in args.mvacc_calibration_thresholds.split(",") if x.strip()]:
            candidate = copy.copy(args)
            features = build_round_features(X_cal, Z_cal, candidate, cflcg_mode="local")
            minimum = max(2, int(round(ratio * len(y_cal))))
            labels, _ = run_hdbscan(features["graph"], minimum, candidate.mvacc_min_samples)
            labels, _ = iterative_merge_clusters_no_drop(labels, features, candidate,
                                                         max_rounds=candidate.mvacc_merge_rounds,
                                                         threshold=threshold)
            labels, _ = assign_noise_samples_no_drop(labels, features, candidate)
            metrics = clustering_metrics(y_cal, labels)
            rows.append({"Candidate Min Cluster Ratio": ratio, "Candidate Merge Threshold": threshold,
                         "Cluster Count Error": abs(int(metrics["Clusters"]) - len(np.unique(y_cal))),
                         "NMI": metrics["NMI"], "Hungarian Acc": metrics["Hungarian Acc"]})
    table = pd.DataFrame(rows).sort_values(["Cluster Count Error", "Hungarian Acc", "NMI"],
                                           ascending=[True, False, False])
    best = table.iloc[0]
    args.mvacc_min_cluster_ratio = float(best["Candidate Min Cluster Ratio"])
    args.mvacc_merge_threshold = float(best["Candidate Merge Threshold"])
    table.to_csv(os.path.join(args.save_dir, "mvacc_day1_validation_calibration.csv"), index=False)
    print(f"[MV-ACC calibration] Day1 validation only: ratio={args.mvacc_min_cluster_ratio}, "
          f"merge_threshold={args.mvacc_merge_threshold}")

def get_split(splits, name, fallback=None):
    if name in splits:
        return splits[name]["X"], splits[name]["y"], splits[name].get("day", "")
    if fallback is not None:
        return fallback
    raise KeyError(f"Missing split: {name}")


def run_discovery(
    method,
    round_name,
    day_name,
    true_new_classes,
    X_round,
    y_round,
    Z_round,
    args,
    full_method=False,
    iarc_method=False,
    recording_ids=None,
):
    cflcg_mode = cflcg_mode_for_method(method)
    feats = build_round_features(X_round, Z_round, args, cflcg_mode=cflcg_mode)

    base_method = method

    if base_method == "Deep only":
        discovery_feat = feats["deep"]
        proto_feat_type = "deep"
    elif base_method == "RF only":
        discovery_feat = feats["rf"]
        proto_feat_type = "rf"
    else:
        discovery_feat = feats["graph"]
        proto_feat_type = "hybrid"

    mv_acc_method = base_method in {"No CF-LCG (MV-ACC)", "Global CF-LCG (MV-ACC)", "MV-ACC"}
    discovery_backend = str(getattr(args, "discovery_backend", "mvacc")).lower()
    gpcc_method = mv_acc_method and discovery_backend in {
        "gpcc",
        "gpcc_recording",
        "gpcc_recording_consensus",
        "gpcc_transmission_prototype",
    }
    if mv_acc_method and discovery_backend == "oracle":
        if not bool(getattr(args, "oracle_discovery_diagnostic", False)):
            raise ValueError(
                "--discovery_backend oracle is diagnostic-only. "
                "Pass --oracle_discovery_diagnostic to make the upper-bound leakage explicit."
            )
        # Stage61 上界诊断：这里故意把当前 discovery 真值变成完美簇，
        # 只用于定位 LoRa 低分是 discovery 还是后端/表征瓶颈。该路径读取
        # 未知真值，绝不能作为正式方法或论文主结果。
        true_ids = sorted(np.unique(np.asarray(y_round, dtype=np.int64)).tolist())
        true_to_cluster = {int(label): idx for idx, label in enumerate(true_ids)}
        labels_for_metrics = np.asarray([true_to_cluster[int(label)] for label in y_round], dtype=np.int64)
        raw_metrics = clustering_metrics(y_round, labels_for_metrics)
        raw_cluster_count = int(raw_metrics["Clusters"])
        initial_cluster_count = int(raw_cluster_count)
        enrolled_ids, details_final = all_non_noise_as_accepted(labels_for_metrics)
        reliability_map = {int(cid): 1.0 for cid in enrolled_ids}
        final_metrics = clustering_metrics(y_round, labels_for_metrics)
        cluster_row = {
            "Method": method,
            "Round": round_name,
            "Discovery Day": day_name,
            "True New Classes": int(true_new_classes),
            "Samples": int(len(y_round)),
            "Initial Cluster Count": int(initial_cluster_count),
            "Final Cluster Count": int(final_metrics["Clusters"]),
            "Cluster Count Error": abs(int(final_metrics["Clusters"]) - int(true_new_classes)),
            "Over-clustering Ratio": float(final_metrics["Clusters"] / max(1, true_new_classes)),
            "Assignment Coverage": float(np.mean(labels_for_metrics >= 0)),
            "Noise Points": int(np.sum(labels_for_metrics < 0)),
            **final_metrics,
            "Discovery Backend": "oracle_diagnostic",
            "Oracle Discovery Diagnostic": True,
            "Oracle Discovery Warning": "uses current discovery true labels; upper bound only",
            "GPCC Uses HDBSCAN": False,
        }
        info = {
            "labels": labels_for_metrics,
            "accepted_ids": enrolled_ids,
            "details": details_final,
            "features": feats,
            "proto_feat_type": proto_feat_type,
            "reliability_map": reliability_map,
            "discovered_clusters": int(final_metrics["Clusters"]),
        }
        save_discovery_visualizations(method, round_name, day_name, discovery_feat, y_round, labels_for_metrics, enrolled_ids, args, full_method)
        return cluster_row, info
    if gpcc_method:
        if discovery_backend in {
            "gpcc_recording",
            "gpcc_recording_consensus",
            "gpcc_transmission_prototype",
        }:
            # LoRa 的同一次物理 transmission 可观测地包含多个 aligned
            # symbol。gpcc_recording 是 Stage 24 的组级平均版本；
            # gpcc_recording_consensus 则先保留 symbol 级 GPCC，再只对组内
            # 高一致 recording 做多数共识修正，避免过度平均压掉 LoRa 细节。
            # WiSig/ADS-B 没有该元数据时直接报错。
            if recording_ids is None:
                raise ValueError(
                    f"{discovery_backend} requires observable recording_ids for the current discovery split."
                )
            if discovery_backend == "gpcc_recording_consensus":
                gpcc = run_recording_consensus_gpcc(
                    feats,
                    recording_ids=np.asarray(recording_ids),
                    target_clusters=int(true_new_classes),
                    seed=int(args.seed),
                    consensus_threshold=float(args.recording_consensus_threshold),
                )
                backend_name = "Recording-Consensus-GPCC"
            elif discovery_backend == "gpcc_transmission_prototype":
                gpcc = run_transmission_prototype_gpcc(
                    feats,
                    recording_ids=np.asarray(recording_ids),
                    target_clusters=int(true_new_classes),
                    seed=int(args.seed),
                )
                backend_name = "Transmission-Prototype-GPCC"
            else:
                gpcc = run_recording_gpcc(
                    feats,
                    recording_ids=np.asarray(recording_ids),
                    target_clusters=int(true_new_classes),
                    seed=int(args.seed),
                )
                backend_name = "Recording-GPCC"
        else:
            gpcc = run_gpcc(feats, target_clusters=int(true_new_classes), seed=int(args.seed))
            backend_name = "GPCC"
        labels_for_metrics = gpcc.labels.astype(np.int64)
        probs = gpcc.confidence.astype(np.float32)
        raw_metrics = clustering_metrics(y_round, labels_for_metrics)
        raw_cluster_count = int(raw_metrics["Clusters"])
        initial_cluster_count = int(raw_cluster_count)
        enrolled_ids, details_final = all_non_noise_as_accepted(labels_for_metrics)
        reliability_map = {
            int(d["cluster_id"]): float(d.get("reliability_score", 1.0))
            for d in details_final
        }
        final_metrics = clustering_metrics(y_round, labels_for_metrics)
        cluster_row = {
            "Method": method,
            "Round": round_name,
            "Discovery Day": day_name,
            "True New Classes": int(true_new_classes),
            "Samples": int(len(y_round)),
            "Initial Cluster Count": int(initial_cluster_count),
            "Final Cluster Count": int(final_metrics["Clusters"]),
            "Cluster Count Error": int(abs(int(final_metrics["Clusters"]) - int(true_new_classes))),
            "Over-clustering Ratio": float(final_metrics["Clusters"] / max(int(true_new_classes), 1)),
            "Assignment Coverage": 1.0,
            "Noise Points": 0,
            **final_metrics,
            "Discovery Backend": backend_name,
            **feats.get("feature_adapter_diagnostics", {}),
            **gpcc.diagnostics,
        }
        info = {
            "labels": labels_for_metrics,
            "accepted_ids": enrolled_ids,
            # 共享 CIL baseline 统一读取 enrolled_ids；GPCC 也要提供同一
            # discovery info 契约，避免主方法能跑而 DOI-style/iCaRL 对照
            # 在伪标签准备阶段因字段名不一致失败。
            "enrolled_ids": enrolled_ids,
            "reliability": reliability_map,
            "merge_rows": [],
            "features": feats,
            "proto_feat_type": proto_feat_type,
            "raw_hdbscan_probabilities": probs,
            "raw_noise_mask": np.zeros(len(labels_for_metrics), dtype=bool),
            "discovered_clusters": int(final_metrics["Clusters"]),
            "enrolled_clusters": int(len(enrolled_ids)),
            "cluster_pseudo_pairs": [],
            "gpcc_diagnostics": gpcc.diagnostics,
        }
        info["discovery_features"] = discovery_feat
        return cluster_row, info

    if mv_acc_method:
        # Scale the density prior with batch size.  The ratio and the remaining
        # hyperparameters are calibrated on Day1 known-class validation data,
        # then frozen before any unknown-round labels are evaluated.
        discovery_min_cluster_size = max(
            10, int(round(float(args.mvacc_min_cluster_ratio) * len(discovery_feat)))
        )
        discovery_min_samples = int(args.mvacc_min_samples)
    else:
        discovery_min_cluster_size = int(args.min_cluster_size)
        discovery_min_samples = int(args.min_samples)

    raw_labels, probs = run_hdbscan(
        discovery_feat, discovery_min_cluster_size, discovery_min_samples
    )
    raw_metrics = clustering_metrics(y_round, raw_labels)
    raw_cluster_count = raw_metrics["Clusters"]
    initial_cluster_count = int(raw_cluster_count)
    iarc_stats = {}

    iter_merge_method = base_method.startswith("IterMerge")
    nodrop_full_method = base_method == "No-drop consolidation (ablation)"

    if mv_acc_method:
        # Multi-view Adaptive Cluster Consolidation (MV-ACC):
        #   1) graph-view HDBSCAN produces density micro-clusters;
        #   2) every noise sample is assigned by deep/RF/graph consensus;
        #   3) mutually-nearest micro-clusters are iteratively consolidated.
        # No sample or cluster is rejected.
        mvacc_args = copy.copy(args)
        mvacc_args.iter_merge_lambda_deep = float(args.mvacc_lambda_deep)
        mvacc_args.iter_merge_lambda_rf = float(args.mvacc_lambda_rf)
        mvacc_args.iter_merge_lambda_graph = float(args.mvacc_lambda_graph)
        mvacc_args.nodrop_assign_lambda_deep = float(args.mvacc_lambda_deep)
        mvacc_args.nodrop_assign_lambda_rf = float(args.mvacc_lambda_rf)
        mvacc_args.nodrop_assign_lambda_graph = float(args.mvacc_lambda_graph)
        mvacc_args.no_mutual_nearest = False

        labels_for_metrics, assignment_stats = assign_noise_samples_no_drop(
            raw_labels, feats, mvacc_args
        )
        labels_for_metrics, merge_stats = iterative_merge_clusters_no_drop(
            labels_for_metrics,
            feats,
            mvacc_args,
            max_rounds=args.mvacc_merge_rounds,
            threshold=args.mvacc_merge_threshold,
        )
        labels_for_metrics, split_stats = adaptive_split_large_clusters_no_drop(
            labels_for_metrics, feats, mvacc_args
        )
        target_stats = {}
        if bool(args.mvacc_enforce_round_size):
            labels_for_metrics, target_stats = merge_closest_clusters_to_target_no_drop(
                labels_for_metrics, feats, mvacc_args, target_count=args.round_size
            )
        iarc_stats = {
            **merge_stats,
            **assignment_stats,
            **split_stats,
            **target_stats,
            "MVACC Min Cluster Size": int(discovery_min_cluster_size),
            "MVACC Min Cluster Ratio": float(args.mvacc_min_cluster_ratio),
            "MVACC Min Samples": int(discovery_min_samples),
        }
        enrolled_ids, details_final = all_non_noise_as_accepted(labels_for_metrics)
        merge_rows = []
    elif full_method:
        accepted_raw, _ = reliability_filter(
            discovery_feat, raw_labels, probs,
            args.reliability_min_cluster_size,
            args.reliability_min_prob,
            args.reliability_threshold,
        )
        final_labels, merge_rows = merge_accepted_clusters(
            raw_labels, accepted_raw, feats["graph"], feats["rf"],
            args.merge_threshold, mutual_nearest=not args.no_mutual_nearest,
        )
        accepted_final, details_final = reliability_filter(
            discovery_feat, final_labels, probs,
            args.reliability_min_cluster_size,
            args.reliability_min_prob,
            args.reliability_threshold,
        )

        iarc_stats = {}
        if iarc_method:
            final_labels, accepted_final, iarc_stats = intra_round_adaptive_reclustering(
                final_labels,
                accepted_final,
                discovery_feat,
                feats,
                args,
            )

        # Recompute final cluster reliability after merge / MV-IARC for optional diagnostics.
        _, details_final = reliability_filter(
            discovery_feat,
            final_labels,
            probs,
            args.reliability_min_cluster_size,
            args.reliability_min_prob,
            threshold=0.0,
        )

        enrolled_ids = accepted_final
        labels_for_metrics = final_labels
    elif nodrop_full_method:
        # Consolidate over-clustered sub-clusters and then reassign every noise
        # sample. No reliability filtering and no sample rejection are used.
        merge_rows = []
        labels_for_metrics, merge_stats = iterative_merge_clusters_no_drop(
            raw_labels, feats, args,
            max_rounds=args.nodrop_merge_rounds,
            threshold=args.nodrop_merge_threshold,
        )
        labels_for_metrics, assignment_stats = assign_noise_samples_no_drop(labels_for_metrics, feats, args)
        iarc_stats = {**merge_stats, **assignment_stats}
        enrolled_ids, details_final = all_non_noise_as_accepted(labels_for_metrics)
    else:
        merge_rows = []
        if iter_merge_method:
            merge_rounds = int(base_method.replace("IterMerge-", ""))
            labels_for_metrics, iter_stats = iterative_merge_clusters_no_drop(
                raw_labels, feats, args, max_rounds=merge_rounds
            )
            enrolled_ids, details_final = all_non_noise_as_accepted(labels_for_metrics)
            iarc_stats.update(iter_stats)
            raw_cluster_count = clustering_metrics(y_round, labels_for_metrics)["Clusters"]
        else:
            enrolled_ids, details_final = all_non_noise_as_accepted(raw_labels)
            labels_for_metrics = raw_labels

    reliability_map = {int(d["cluster_id"]): float(d.get("reliability_score", 1.0)) for d in details_final}
    final_metrics = clustering_metrics(y_round, labels_for_metrics)

    cluster_row = {
        "Method": method,
        "Round": round_name,
        "Discovery Day": day_name,
        "True New Classes": int(true_new_classes),
        "Samples": int(len(y_round)),
        "Initial Cluster Count": int(initial_cluster_count),
        "Final Cluster Count": int(final_metrics["Clusters"]),
        "Cluster Count Error": int(abs(int(final_metrics["Clusters"]) - int(true_new_classes))),
        "Over-clustering Ratio": float(final_metrics["Clusters"] / max(int(true_new_classes), 1)),
        **final_metrics,
        **feats.get("feature_adapter_diagnostics", {}),
    }

    graph_based = base_method in [
        "Graph fusion (raw)",
        "No-drop consolidation (ablation)",
        "No CF-LCG (MV-ACC)",
        "Global CF-LCG (MV-ACC)",
        "MV-ACC",
        "Full method",
        "MV-IARC",
    ]
    if graph_based and args.adaptive_fusion:
        cluster_row.update({
            "Fusion": "bounded_adaptive",
            "Alpha Min": float(args.alpha_min),
            "Alpha Max": float(args.alpha_max),
            **feats.get("fusion_stats", {}),
        })
    elif graph_based:
        cluster_row.update({
            "Fusion": "fixed_alpha",
            "Alpha": float(args.alpha),
        })

    if (iarc_method or iter_merge_method or nodrop_full_method or mv_acc_method) and iarc_stats:
        cluster_row.update(iarc_stats)

    # Paper-facing diagnostics: save every method and every discovery round,
    # not only the final method.
    if args.enable_visualization and args.save_detailed_visualizations:
        save_discovery_visualizations(
        method=method,
        round_name=round_name,
        day_name=day_name,
        features=discovery_feat,
        y_true=y_round,
        cluster_labels=labels_for_metrics,
        accepted_ids=enrolled_ids,
        args=args,
            full_method=(full_method or iarc_method or nodrop_full_method or mv_acc_method),
        )

    info = {
        "method": method,
        "round": round_name,
        "proto_feat_type": proto_feat_type,
        "labels": labels_for_metrics,
        "raw_hdbscan_probabilities": np.asarray(probs if probs is not None else np.ones(len(labels_for_metrics)), dtype=np.float32),
        "raw_noise_mask": np.asarray(raw_labels == -1, dtype=bool),
        "enrolled_ids": enrolled_ids,
        "discovered_clusters": int(final_metrics["Clusters"] if (nodrop_full_method or mv_acc_method) else raw_cluster_count),
        "enrolled_clusters": int(len(enrolled_ids)),
        "merge_count": int(iarc_stats.get("Iter Merge Total Merges", len(merge_rows))),
        "iarc_stats": iarc_stats,
        "reliability_map": reliability_map,
    }

    info["discovery_features"] = discovery_feat
    return cluster_row, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_profile", choices=["wisig", "lora25"], default="wisig", help="Dataset-specific strict split profile; the algorithmic MV-ACC-CIL core is shared.")
    parser.add_argument("--dataset_path", type=str, default=r"D:\WiSigCustom\WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl")
    parser.add_argument("--save_dir", type=str, default="./results/wisig_rx2_10known_3round_nodrop")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--train_closedset", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument(
        "--closedset_backbone",
        choices=["resnet1d", "lora_chirp", "lora_hybrid"],
        default="resnet1d",
        help="Initial closed-set backbone. lora_hybrid fuses raw IQ and magnitude/phase geometry views.",
    )
    parser.add_argument("--use_supcon", action="store_true", help="Use CE + supervised contrastive loss when training the initial closed-set backbone.")
    parser.add_argument("--supcon_weight", type=float, default=0.1, help="Weight lambda for supervised contrastive loss. Recommended: 0.05 or 0.1.")
    parser.add_argument("--supcon_temperature", type=float, default=0.2, help="Temperature for supervised contrastive loss. Recommended: 0.2.")
    parser.add_argument("--closedset_val_ratio", type=float, default=1.0 / 7.0, help="Validation fraction inside the Day1 70%% training portion; 1/7 yields an overall 60/10/30 split.")
    parser.add_argument("--projection_hidden_dim", type=int, default=128)
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--disable_rf_augmentation", action="store_true")
    parser.add_argument("--disable_cflcg", action="store_true", help="Disable Day1-validation CF-LCG classical-feature selection.")
    parser.add_argument("--cflcg_gate_threshold", type=float, default=0.80)
    parser.add_argument("--cflcg_local_k", type=int, default=12, help="kNN size for label-free Local CF-LCG cross-view consistency.")
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--initial_known_classes", type=int, default=10)
    parser.add_argument("--round_size", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument(
        "--discovery_backend",
        choices=[
            "mvacc",
            "gpcc",
            "gpcc_recording",
            "gpcc_recording_consensus",
            "gpcc_transmission_prototype",
            "oracle",
        ],
        default="mvacc",
        help="Discovery front-end: gpcc fixes K=round_size; oracle is diagnostic-only and requires --oracle_discovery_diagnostic.",
    )
    parser.add_argument("--oracle_discovery_diagnostic", action="store_true", help="允许 discovery_backend=oracle 读取当前 discovery 真值做上界诊断；该结果不能作为正式方法。")
    parser.add_argument(
        "--recording_consensus_threshold",
        type=float,
        default=0.75,
        help="gpcc_recording_consensus 中 recording 内多数簇占比达到该阈值才统一伪标签；只使用 recording_id 和无标签聚类结果。",
    )
    parser.add_argument(
        "--enable_joint_discovery_refinement",
        action="store_true",
        help="每轮先完成一次 CIL，再用 Student 表征重新发现并无标签对齐，最后进行第二次 CIL；默认关闭。",
    )
    parser.add_argument(
        "--radcil_refined_filter_top_fraction",
        type=float,
        default=1.0,
        help="Joint refinement 第二次 CIL 仅保留 Student/二次伪标签一致的类内 top fraction；1.0 保持历史全量二次训练。",
    )
    parser.add_argument(
        "--radcil_refined_filter_min_per_class",
        type=int,
        default=1,
        help="Joint refinement 高置信一致过滤时每个二次伪类至少保留的样本数；仅在 top_fraction < 1 时生效。",
    )
    parser.add_argument("--discovery_feature_adapter", choices=["none", "mn_smooth", "proto_repulse"], default="none", help="Strict discovery feature adapter before MV-ACC/GPCC; uses only known train and current discovery features.")
    parser.add_argument("--cross_day_repr_adaptation", action="store_true", help="每轮 Teacher discovery 前做训练期跨天表征适配，只使用 Day1 known train 标签和当前 discovery 无标签样本。")
    parser.add_argument("--cross_day_repr_epochs", type=int, default=2)
    parser.add_argument("--cross_day_repr_batch_size", type=int, default=128)
    parser.add_argument("--cross_day_repr_lr", type=float, default=1e-5)
    parser.add_argument("--cross_day_repr_consistency_weight", type=float, default=0.5)
    parser.add_argument("--cross_day_repr_coral_weight", type=float, default=0.05)
    parser.add_argument(
        "--lora_old_day_calibration_adaptation",
        action="store_true",
        help="LoRa 专用训练期跨天旧类监督适配；读取当前日期旧设备 IQ_1-7 标注样本，不读取 IQ_8-10。",
    )
    parser.add_argument(
        "--lora_old_day_calibration_raw_dir",
        type=str,
        default=None,
        help="LoRa Setup 1 原始 I/Q 根目录，需包含 Day2/Day3/Day4/Device*/IQ_*.dat。",
    )
    parser.add_argument("--lora_old_day_calibration_transmissions", type=str, default="1,2,3,4,5,6,7")
    parser.add_argument("--lora_old_day_calibration_symbols", type=int, default=28)
    parser.add_argument("--lora_old_day_calibration_decimation", type=int, default=1)
    parser.add_argument("--lora_old_day_calibration_representation", choices=["raw", "dechirped"], default="raw")
    parser.add_argument("--lora_old_day_calibration_epochs", type=int, default=2)
    parser.add_argument("--lora_old_day_calibration_batch_size", type=int, default=96)
    parser.add_argument("--lora_old_day_calibration_lr", type=float, default=2e-5)
    parser.add_argument("--lora_old_day_calibration_ce_weight", type=float, default=1.0)
    parser.add_argument("--lora_old_day_calibration_known_ce_weight", type=float, default=0.25)
    parser.add_argument("--lora_old_day_calibration_feature_weight", type=float, default=0.5)
    parser.add_argument("--lora_old_day_calibration_anchor_weight", type=float, default=0.1)
    parser.add_argument(
        "--lora_old_day_joint_cil",
        action="store_true",
        help="LoRa 专用：把当前日期旧设备 IQ_1-7 标注样本直接并入每轮 CIL，"
        "与当前伪新类和 replay 旧类联合优化；默认关闭。",
    )
    parser.add_argument(
        "--lora_old_day_joint_cil_batch_size",
        type=int,
        default=96,
        help="每个 CIL 优化步抽取的旧类跨天校准 batch 大小。",
    )
    parser.add_argument(
        "--lora_old_day_joint_cil_ce_weight",
        type=float,
        default=1.0,
        help="旧类跨天校准监督 CE 权重。",
    )
    parser.add_argument(
        "--lora_old_day_joint_cil_feature_weight",
        type=float,
        default=0.5,
        help="旧类跨天校准 batch 与 CIL 开始前 Teacher 的特征方向对齐权重。",
    )
    parser.add_argument("--lora_ssl_adaptation", action="store_true", help="每轮 LoRa discovery 前做无标签 instance contrastive 自监督适配；默认关闭。")
    parser.add_argument("--lora_ssl_epochs", type=int, default=3)
    parser.add_argument("--lora_ssl_batch_size", type=int, default=128)
    parser.add_argument("--lora_ssl_lr", type=float, default=1e-5)
    parser.add_argument("--lora_ssl_ce_weight", type=float, default=1.0)
    parser.add_argument("--lora_ssl_instance_weight", type=float, default=0.5)
    parser.add_argument("--lora_ssl_teacher_weight", type=float, default=0.2)
    parser.add_argument("--lora_ssl_temperature", type=float, default=0.2)
    parser.add_argument("--lora_recording_ssl_adaptation", action="store_true", help="每轮 LoRa discovery 前使用 recording_id 做 must-link 自监督适配；默认关闭。")
    parser.add_argument("--lora_recording_ssl_epochs", type=int, default=3)
    parser.add_argument("--lora_recording_ssl_batch_size", type=int, default=128)
    parser.add_argument("--lora_recording_ssl_lr", type=float, default=1e-5)
    parser.add_argument("--lora_recording_ssl_ce_weight", type=float, default=1.0)
    parser.add_argument("--lora_recording_ssl_recording_weight", type=float, default=0.5)
    parser.add_argument("--lora_recording_ssl_instance_weight", type=float, default=0.2)
    parser.add_argument("--lora_recording_ssl_teacher_weight", type=float, default=0.2)
    parser.add_argument("--lora_recording_ssl_temperature", type=float, default=0.2)
    parser.add_argument("--discovery_adapter_smooth_k", type=int, default=12, help="kNN size for discovery-only local feature smoothing.")
    parser.add_argument("--discovery_adapter_smooth_weight", type=float, default=0.20, help="Blend weight for discovery-only local feature smoothing.")
    parser.add_argument("--discovery_adapter_repulsion_weight", type=float, default=0.15, help="Known-prototype repulsion weight for proto_repulse adapter.")
    parser.add_argument("--discovery_only", action="store_true", help="只运行严格发现前端并保存聚类表；用于先验证聚类质量，不进入增量训练后端。")
    parser.add_argument("--enable_lora_recording_eval", action="store_true", help="LoRa held-out eval 额外输出 recording_id 多数投票诊断，不改变训练或正式 symbol-level 指标。")
    parser.add_argument("--save_end_to_end_eval_dumps", action="store_true", help="保存端到端 held-out eval logits/pred/mapping，用于离线组合路由诊断；默认关闭。")
    parser.add_argument("--development_ratio", type=float, default=0.70, help="Per-day development ratio: Day1 becomes 60%% backbone training + 10%% validation; remaining 30%% is held-out evaluation.")
    parser.add_argument(
        "--selected_rx_list",
        type=str,
        default="2",
        help="Fixed receiver index. This RX2 experiment only accepts 2 (the third receiver).",
    )
    parser.add_argument("--old_class_bonus", type=float, default=0.0, help="Old-class prototype score bonus beta for reducing forgetting. Recommended: 0.00, 0.02, 0.04, 0.06.")
    parser.add_argument("--disable_cil_baselines", action="store_true", help="Disable representative class-incremental baselines (Ft-CNN/LwF/iCaRL/EEIL/TPCIL-style/DOI-style).")
    parser.add_argument("--cil_epochs", type=int, default=12, help="Epochs for lightweight frozen-feature CIL baselines.")
    parser.add_argument("--cil_batch_size", type=int, default=256, help="Batch size for frozen-feature CIL baselines.")
    parser.add_argument("--cil_lr", type=float, default=1e-3, help="Learning rate for frozen-feature CIL baselines.")
    parser.add_argument("--cil_weight_decay", type=float, default=1e-4, help="Weight decay for frozen-feature CIL baselines.")
    parser.add_argument("--cil_distill_weight", type=float, default=1.0, help="Knowledge distillation weight for LwF/EEIL baselines.")
    parser.add_argument("--cil_distill_temperature", type=float, default=2.0, help="Knowledge distillation temperature for LwF/EEIL baselines.")
    parser.add_argument("--memory_per_class", type=int, default=20, help="Number of exemplars per class for iCaRL/EEIL/TPCIL-style/DOI-style baselines.")
    parser.add_argument("--tpcil_smoothing", type=float, default=0.20, help="Prototype graph smoothing strength for TPCIL-style baseline.")
    parser.add_argument("--cil_head_warmup_epochs", type=int, default=2, help="End-to-end MV-ACC-CIL head-only warm-up epochs.")
    parser.add_argument("--cil_joint_epochs", type=int, default=8, help="End-to-end MV-ACC-CIL last-block joint fine-tuning epochs.")
    parser.add_argument("--incremental_batch_size", type=int, default=128)
    parser.add_argument("--cil_classifier_lr", type=float, default=1e-4)
    parser.add_argument("--cil_backbone_lr", type=float, default=1e-5)
    parser.add_argument("--cil_replay_weight", type=float, default=1.0)
    parser.add_argument("--cil_kd_weight", type=float, default=1.0)
    parser.add_argument("--cil_temperature", type=float, default=2.0)
    parser.add_argument("--cil_supcon_weight", type=float, default=0.05)
    parser.add_argument("--cil_supcon_threshold", type=float, default=0.60)
    parser.add_argument(
        "--radcil_metric_weight",
        type=float,
        default=0.0,
        help="Incremental normalized cosine-proxy metric loss weight; 0 preserves historical RADCIL behavior.",
    )
    parser.add_argument(
        "--radcil_metric_scale",
        type=float,
        default=16.0,
        help="Scale applied to normalized cosine-proxy logits.",
    )
    parser.add_argument(
        "--radcil_metric_margin",
        type=float,
        default=0.05,
        help="True-class cosine margin for the metric loss; must be in [0, 1).",
    )
    parser.add_argument("--radcil_old_new_batch_ratio", type=float, default=0.0, help="RADCIL replay old:new batch ratio; 0 keeps the legacy equal batch-size behavior.")
    parser.add_argument(
        "--radcil_balanced_head_recalibration_epochs",
        type=int,
        default=0,
        help="冻结backbone后按类均衡重校准RADCIL分类头的轮数；0保持历史行为。",
    )
    parser.add_argument(
        "--radcil_balanced_head_new_distill_weight",
        type=float,
        default=0.0,
        help="类均衡分类头重校准时对当前轮新类logits保持教师约束的权重；0保持Stage31行为。",
    )
    parser.add_argument("--radcil_masked_kd", action="store_true", help="Apply KD only on replay samples whose labels are inside the teacher output range.")
    parser.add_argument("--radcil_kd_schedule", choices=["constant", "cosine", "linear_decay"], default="constant", help="Schedule for the end-to-end CIL KD weight inside each training stage.")
    parser.add_argument("--radcil_feature_distill_weight", type=float, default=0.0, help="Replay-feature distillation weight for constraining old-class backbone drift.")
    parser.add_argument(
        "--radcil_old_replay_supcon_weight",
        type=float,
        default=0.0,
        help="Replay-only old-class supervised contrastive weight; 0 preserves historical RADCIL behavior.",
    )
    parser.add_argument("--radcil_prototype_anchor_weight", type=float, default=0.0, help="Teacher replay class-prototype anchor weight; 0 preserves historical RADCIL behavior.")
    parser.add_argument(
        "--radcil_aug_consistency_weight",
        type=float,
        default=0.0,
        help="Incremental dual-view feature consistency weight; 0 preserves historical RADCIL behavior.",
    )
    parser.add_argument(
        "--radcil_domain_alignment_weight",
        type=float,
        default=0.0,
        help="Cross-day CORAL-style feature alignment weight between current discovery and replay; 0 preserves historical RADCIL behavior.",
    )
    parser.add_argument(
        "--pseudo_weight_use_cluster_reliability",
        action="store_true",
        help="Multiply pseudo-label weights by discovery-side cluster reliability; disabled by default.",
    )
    parser.add_argument(
        "--pseudo_weight_cluster_reliability_floor",
        type=float,
        default=0.20,
        help="Minimum pseudo-label weight after cluster-reliability scaling.",
    )
    parser.add_argument("--radcil_unfreeze_scope", choices=["none", "fc", "layer3", "tail", "layer2_tail"], default="tail", help="Backbone scope unfrozen during the joint RADCIL stage.")
    parser.add_argument("--radcil_doi_fusion_weight", type=float, default=0.0, help="DOI replay-prototype late-fusion weight; 0 keeps network-only RADCIL prediction.")
    parser.add_argument("--radcil_doi_align_lambda", type=float, default=0.30, help="Current-round weight used to align historical and current replay prototypes.")
    parser.add_argument("--radcil_doi_prototype_temperature", type=float, default=0.10, help="Temperature applied to cosine prototype scores before late fusion.")
    parser.add_argument("--radcil_grouped_doi_fusion", action="store_true", help="Fuse DOI prototypes only into old-class columns; current-round new classes retain network-head scores.")
    parser.add_argument("--radcil_doi_fusion_candidates", type=str, default="0,0.25,0.5,0.75,1.0", help="Day1-validation candidates for grouped DOI old-class fusion.")
    parser.add_argument("--radcil_old_prototype_route", action="store_true", help="Route high old-mass predictions through replay prototypes for old-class internal decisions.")
    parser.add_argument("--radcil_old_prototype_route_candidates", type=str, default="0.45,0.55,0.65,0.75,0.85", help="Day1/IQ_7 validation candidates for conservative old-prototype routing.")
    parser.add_argument("--radcil_icarl_fallback", action="store_true", help="Enable confidence-gated iCaRL exemplar fallback during evaluation.")
    parser.add_argument("--radcil_icarl_fallback_threshold", type=float, default=0.60, help="Confidence threshold for exemplar fallback when validation-based calibration is not used.")
    parser.add_argument("--radcil_icarl_fallback_candidates", type=str, default="0.45,0.55,0.65,0.75,0.85", help="Day1-validation candidates for confidence-gated exemplar fallback.")
    parser.add_argument("--radcil_old_logit_bias_candidates", type=str, default="", help="Comma-separated old-class logit bias candidates calibrated on Day1/IQ_7 validation; empty disables this diagnostic backend.")
    parser.add_argument("--radcil_bn_recalibration", action="store_true", help="Refresh BatchNorm statistics with replay/current discovery data before discovery and evaluation.")
    parser.add_argument("--radcil_bn_recalibration_passes", type=int, default=1, help="Number of no-grad passes used for BatchNorm statistic recalibration.")
    parser.add_argument("--radcil_bn_recalibration_reset", action="store_true", help="Reset BatchNorm running stats before recalibration; default keeps source-domain stats as a prior.")
    parser.add_argument("--pseudo_weight_floor", type=float, default=0.20)
    parser.add_argument(
        "--radcil_registration_top_fraction",
        type=float,
        default=1.0,
        help="只用于新类 imprint 和 replay 注册的每类高置信样本比例；当前轮完整训练仍使用全部 discovery 样本。",
    )
    parser.add_argument(
        "--radcil_registration_min_per_class",
        type=int,
        default=1,
        help="每个伪类至少保留多少样本用于 imprint/replay 注册。",
    )
    parser.add_argument(
        "--radcil_new_prototype_weight",
        type=float,
        default=0.0,
        help="当前轮伪类原型归属损失权重；0保持历史训练路径。",
    )
    parser.add_argument(
        "--radcil_new_imprint_scale",
        type=float,
        default=1.0,
        help="新伪类 classifier imprint 的权重尺度倍率；1.0 保持历史初始化。",
    )
    parser.add_argument(
        "--radcil_new_imprint_bias",
        type=float,
        default=0.0,
        help="新伪类 classifier imprint 的 bias 初值；0.0 保持历史初始化。",
    )
    parser.add_argument(
        "--radcil_new_head_boost",
        type=float,
        default=1.0,
        help="head warmup 阶段当前轮新伪类 CE 的相对权重；1.0 保持历史训练。",
    )
    parser.add_argument(
        "--radcil_pseudo_aug_consistency_weight",
        type=float,
        default=0.0,
        help="当前轮高置信伪标签样本的增强前后logits一致性权重；0保持历史训练路径。",
    )
    parser.add_argument(
        "--radcil_recording_consistency_weight",
        type=float,
        default=0.0,
        help="LoRa 当前轮同一 recording 内同伪类 symbol 的 logits 一致性权重；0保持历史训练路径。",
    )
    parser.add_argument(
        "--radcil_recording_consistency_temperature",
        type=float,
        default=2.0,
        help="LoRa recording logits 一致性 KL 温度，仅在权重大于0时使用。",
    )
    parser.add_argument(
        "--radcil_recording_ce_weight",
        type=float,
        default=0.0,
        help="LoRa 当前轮同 recording、同伪类 symbol 的 mean-logit CE 权重；0保持历史训练路径。",
    )


    parser.add_argument("--min_cluster_size", type=int, default=10)
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--graph_dim", type=int, default=16)
    parser.add_argument("--adaptive_fusion", action="store_true", help="Use bounded edge-wise adaptive graph fusion for graph-based methods.")
    parser.add_argument("--alpha_min", type=float, default=0.65, help="Minimum deep-view edge weight for bounded adaptive fusion.")
    parser.add_argument("--alpha_max", type=float, default=0.95, help="Maximum deep-view edge weight for bounded adaptive fusion.")

    parser.add_argument("--reliability_min_cluster_size", type=int, default=10)
    parser.add_argument("--reliability_min_prob", type=float, default=0.30)
    parser.add_argument("--reliability_threshold", type=float, default=0.50)
    parser.add_argument("--merge_threshold", type=float, default=0.70)
    parser.add_argument("--no_mutual_nearest", action="store_true")

    # MV-IARC: intra-round adaptive re-clustering parameters.
    parser.add_argument("--iarc_secondary_min_cluster_size", type=int, default=50, help="HDBSCAN min_cluster_size for re-clustering uncertain samples in MV-IARC.")
    parser.add_argument("--iarc_secondary_min_samples", type=int, default=5, help="HDBSCAN min_samples for re-clustering uncertain samples in MV-IARC.")
    parser.add_argument("--iarc_min_secondary_size", type=int, default=30, help="Minimum secondary cluster size allowed for MV-IARC merge/new decisions.")
    parser.add_argument("--iarc_merge_threshold", type=float, default=0.78, help="Similarity threshold for merging a secondary cluster into an existing reliable cluster.")
    parser.add_argument("--iarc_new_threshold", type=float, default=0.35, help="Reliability threshold for accepting a secondary cluster as a new pseudo-class.")
    parser.add_argument("--iarc_lambda_deep", type=float, default=0.4, help="Deep-prototype weight in MV-IARC cluster similarity.")
    parser.add_argument("--iarc_lambda_rf", type=float, default=0.2, help="RF-prototype weight in MV-IARC cluster similarity.")
    parser.add_argument("--iarc_lambda_graph", type=float, default=0.4, help="Graph-prototype weight in MV-IARC cluster similarity.")

    # Pure no-drop iterative cluster merging parameters.
    parser.add_argument("--iter_merge_threshold", type=float, default=0.86, help="Prototype similarity threshold for no-drop iterative cluster merging.")
    parser.add_argument("--iter_merge_lambda_deep", type=float, default=0.4, help="Deep-prototype weight in no-drop iterative merging.")
    parser.add_argument("--iter_merge_lambda_rf", type=float, default=0.2, help="RF-prototype weight in no-drop iterative merging.")
    parser.add_argument("--iter_merge_lambda_graph", type=float, default=0.4, help="Graph-prototype weight in no-drop iterative merging.")
    parser.add_argument("--nodrop_merge_rounds", type=int, default=5, help="Maximum rounds for the no-drop consolidation ablation.")
    parser.add_argument("--nodrop_merge_threshold", type=float, default=0.78, help="Multi-view merge threshold for the no-drop consolidation ablation.")
    parser.add_argument("--nodrop_assign_lambda_deep", type=float, default=0.4, help="Deep weight for no-drop noise reassignment.")
    parser.add_argument("--nodrop_assign_lambda_rf", type=float, default=0.2, help="RF weight for no-drop noise reassignment.")
    parser.add_argument("--nodrop_assign_lambda_graph", type=float, default=0.4, help="Graph weight for no-drop noise reassignment.")

    # MV-ACC: calibration-frozen, no-drop multi-view consolidation.
    parser.add_argument("--mvacc_min_cluster_ratio", type=float, default=0.045, help="HDBSCAN minimum-cluster-size ratio, calibrated on Day1 known validation data.")
    parser.add_argument("--mvacc_min_samples", type=int, default=5, help="HDBSCAN min_samples for MV-ACC.")
    parser.add_argument("--mvacc_merge_rounds", type=int, default=12, help="Maximum mutually-nearest consolidation rounds for MV-ACC.")
    parser.add_argument("--mvacc_merge_threshold", type=float, default=0.74, help="Frozen multi-view prototype merge threshold for MV-ACC.")
    parser.add_argument("--disable_mvacc_calibration", action="store_true", help="Use supplied MV-ACC parameters without Day1 validation calibration.")
    parser.add_argument("--mvacc_calibration_ratios", type=str, default="0.03,0.045,0.06,0.08", help="Day1-validation density-ratio candidates.")
    parser.add_argument("--mvacc_calibration_thresholds", type=str, default="0.74,0.78,0.82,0.86", help="Day1-validation merge-threshold candidates.")
    parser.add_argument("--mvacc_lambda_deep", type=float, default=0.5, help="Deep-view weight in MV-ACC reassignment and consolidation.")
    parser.add_argument("--mvacc_lambda_rf", type=float, default=0.1, help="RF-view weight in MV-ACC reassignment and consolidation.")
    parser.add_argument("--mvacc_lambda_graph", type=float, default=0.4, help="Graph-view weight in MV-ACC reassignment and consolidation.")
    parser.add_argument("--mvacc_prototypes_per_class", type=int, default=5, help="Number of K-means sub-prototypes per enrolled class for MV-ACC.")
    parser.add_argument("--mvacc_split_rounds", type=int, default=6, help="Maximum adaptive oversized-cluster splitting rounds.")
    parser.add_argument("--mvacc_split_size_factor", type=float, default=1.75, help="Split-test clusters larger than this multiple of the median cluster size.")
    parser.add_argument("--mvacc_split_min_part_ratio", type=float, default=0.30, help="Minimum child size relative to the median cluster size.")
    parser.add_argument("--mvacc_split_silhouette", type=float, default=0.30, help="Minimum internal silhouette required to retain an adaptive split.")
    parser.add_argument("--mvacc_enforce_round_size", action="store_true", help="Merge over-clustered MV-ACC output to the protocol-declared per-round class count without using discovery labels.")

    parser.add_argument("--enable_visualization", action="store_true", default=True, help="Save 2D discovery visualizations. Default: enabled.")
    parser.add_argument("--disable_visualization", action="store_true", help="Disable visualization output.")
    parser.add_argument("--visualization_method", type=str, default="both", choices=["tsne", "pca", "umap", "both"], help="Legacy diagnostic projection method.")
    parser.add_argument("--save_detailed_visualizations", action="store_true", help="Also save legacy per-method unknown-only plots.")

    args = parser.parse_args()
    if args.disable_visualization:
        args.enable_visualization = False
    set_seed(args.seed)
    ensure_dir(args.save_dir)

    expected_round_size = 5 if args.dataset_profile == "lora25" else 10
    if args.initial_known_classes != 10 or args.round_size != expected_round_size or args.num_rounds != 3:
        raise ValueError(
            f"The {args.dataset_profile} profile requires 10 known + 3 rounds of "
            f"{expected_round_size} classes each."
        )

    selected_rx_list = [int(x.strip()) for x in args.selected_rx_list.split(",") if x.strip() != ""]
    if args.dataset_profile == "wisig" and selected_rx_list != [2]:
        raise ValueError(
            "This RX2-only script requires --selected_rx_list 2 "
            "(Python index 2 is the third receiver)."
        )
    if args.closedset_backbone in {"lora_chirp", "lora_hybrid"} and args.dataset_profile != "lora25":
        raise ValueError("--closedset_backbone lora_chirp/lora_hybrid is only valid for the LoRa25 strict profile.")
    if args.lora_old_day_calibration_adaptation or args.lora_old_day_joint_cil:
        if args.dataset_profile != "lora25":
            raise ValueError(
                "--lora_old_day_calibration_adaptation/--lora_old_day_joint_cil "
                "are only supported for the LoRa25 strict profile."
            )
        if not args.lora_old_day_calibration_raw_dir:
            raise ValueError(
                "--lora_old_day_calibration_raw_dir is required when old-day "
                "calibration is enabled."
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.checkpoint is None:
        checkpoint_name = (
            f"closedset_lora25_{args.closedset_backbone}_10known.pth"
            if args.dataset_profile == "lora25"
            else "closedset_rx2_day1_10known.pth"
        )
        args.checkpoint = os.path.join(args.save_dir, checkpoint_name)

    dataset_title = "LoRa25 Different Days Indoor" if args.dataset_profile == "lora25" else "WiSig Cross-Day"
    print(f"\n========== {dataset_title} 10-Known 3-Round Incremental Experiment ==========")
    print(f"Dataset: {args.dataset_path}")
    print(f"Save dir: {args.save_dir}")
    print(f"Device: {device}")
    print(f"Protocol: 10 known + {args.round_size} unknown R1 + {args.round_size} unknown R2 + {args.round_size} unknown R3")
    if args.dataset_profile == "wisig":
        print("Receiver setting: selected_rx_list=[2] | mode=single-Rx | physical receiver=third")
    else:
        print("LoRa split: IQ_1-6 train | IQ_7 validation | IQ_8-10 held-out evaluation")
    print(f"Old-class prototype bonus beta: {args.old_class_bonus}")
    print(
        "Incremental dual-view consistency weight: "
        f"{args.radcil_aug_consistency_weight}"
    )
    print(
        "Cross-day feature domain alignment weight: "
        f"{args.radcil_domain_alignment_weight}"
    )
    print(
        "Old replay SupCon weight: "
        f"{args.radcil_old_replay_supcon_weight}"
    )
    print(
        "Pseudo-label cluster reliability weighting: "
        f"{args.pseudo_weight_use_cluster_reliability} "
        f"(floor={args.pseudo_weight_cluster_reliability_floor})"
    )
    if args.use_supcon:
        print(f"Closed-set training loss: CE + {args.supcon_weight} * SupCon(T={args.supcon_temperature})")
    else:
        print("Closed-set training loss: CE only")
    print(f"Sample-level protocol: {args.development_ratio * 6 / 7:.0%} backbone train + {args.development_ratio / 7:.0%} validation + {1.0 - args.development_ratio:.0%} held-out evaluation")
    if args.adaptive_fusion:
        print(f"Fusion: bounded edge-wise adaptive | alpha_min={args.alpha_min}, alpha_max={args.alpha_max}")
    else:
        print(f"Fusion: fixed alpha | alpha={args.alpha}")
    print(
        f"No-drop consolidation (ablation): rounds={args.nodrop_merge_rounds}, "
        f"threshold={args.nodrop_merge_threshold}, coverage=100%"
    )
    print(
        f"MV-ACC: mcs_ratio={args.mvacc_min_cluster_ratio}, min_samples={args.mvacc_min_samples}, "
        f"merge_th={args.mvacc_merge_threshold}, sub_prototypes={args.mvacc_prototypes_per_class}, "
        f"split_factor={args.mvacc_split_size_factor}, split_sil={args.mvacc_split_silhouette}, "
        f"coverage=100%, strict_alignment=Hungarian"
    )

    if args.dataset_profile == "lora25":
        splits = load_lora25_diffdays_3round(args.dataset_path)
        train_split_key = "day1_backbone_train"
        validation_split_key = "day1_known_validation"
    else:
        splits = load_wisig_crossday_10known_3round(
            dataset_path=args.dataset_path,
            transpose_to_model=True,
            train_ratio=args.development_ratio,
            selected_rx_list=selected_rx_list,
            seed=args.seed,
        )
        train_split_key = "day1_known_train"
        validation_split_key = None

    X_train = splits[train_split_key]["X"]
    y_train = splits[train_split_key]["y"]
    X_validation = splits[validation_split_key]["X"] if validation_split_key else None
    y_validation = splits[validation_split_key]["y"] if validation_split_key else None

    round_keys = [
        "day2_unknown_round1",
        "day3_unknown_round2",
        "day4_unknown_round3",
    ]
    round_data = []
    for rk in round_keys:
        round_data.append({
            "key": rk,
            "X": splits[rk]["X"],
            "y": splits[rk]["y"],
            "day": splits[rk].get("day", ""),
            # LoRa loader 提供 recording_id；其它数据集没有该元数据时保持
            # None，只有 gpcc_recording 会显式要求它存在。
            "recording_id": splits[rk].get("recording_id"),
        })

    eval_keys = {
        "eval_initial": "day1_initial_eval",
        "eval_r1": "day2_eval_after_r1",
        "eval_r2": "day3_eval_after_r2",
        "eval_r3": "day4_eval_after_r3",
    }
    eval_data = {}
    for bank_key, split_key in eval_keys.items():
        eval_data[bank_key] = {
            "X": splits[split_key]["X"],
            "y": splits[split_key]["y"],
            "day": splits[split_key].get("day", ""),
            "split_key": split_key,
            # LoRa strict loader 带有 recording_id。其它数据集没有该元数据时
            # 保持 None，recording-level 诊断会自动跳过。
            "recording_id": splits[split_key].get("recording_id"),
        }

    development_label = "IQ_1-6 train" if args.dataset_profile == "lora25" else "70%"
    discovery_label = "IQ_1-7 discovery" if args.dataset_profile == "lora25" else "70%"
    evaluation_label = "IQ_8-10 eval" if args.dataset_profile == "lora25" else "30%"
    print("\n[Splits]")
    print(f"Day1 known train {development_label}: {X_train.shape}, classes={np.unique(y_train).size}, labels={np.min(y_train)}-{np.max(y_train)}")
    print(f"Initial eval 30%:     {eval_data['eval_initial']['X'].shape}, classes={np.unique(eval_data['eval_initial']['y']).size}, labels={np.min(eval_data['eval_initial']['y'])}-{np.max(eval_data['eval_initial']['y'])}, day={eval_data['eval_initial']['day']}")
    for i, rd in enumerate(round_data, start=1):
        ek = f"eval_r{i}"
        print(f"Unknown R{i} {discovery_label}: {rd['X'].shape}, classes={np.unique(rd['y']).size}, labels={np.min(rd['y'])}-{np.max(rd['y'])}, day={rd['day']}")
        print(f"After R{i} {evaluation_label}: {eval_data[ek]['X'].shape}, classes={np.unique(eval_data[ek]['y']).size}, labels={np.min(eval_data[ek]['y'])}-{np.max(eval_data[ek]['y'])}, day={eval_data[ek]['day']}")

    train_set = to_dataset(X_train, y_train)
    validation_set = to_dataset(X_validation, y_validation) if X_validation is not None else None
    # 主流程只通过这个工厂创建 Day1 closed-set backbone。默认 resnet1d
    # 完全保持历史行为；lora_chirp 和 lora_hybrid 只在 LoRa profile 中显式启用。
    if args.closedset_backbone == "lora_chirp":
        closedset_model_factory = lambda: LoRaChirpClosedSet(
            num_known_classes=args.initial_known_classes,
            feat_dim=args.feat_dim,
        )
    elif args.closedset_backbone == "lora_hybrid":
        closedset_model_factory = lambda: LoRaHybridClosedSet(
            num_known_classes=args.initial_known_classes,
            feat_dim=args.feat_dim,
        )
    else:
        closedset_model_factory = lambda: ClosedSetSEI(
            num_known_classes=args.initial_known_classes,
            feat_dim=args.feat_dim,
        )
    split_protocol_name = (
        "lora25_transmission_disjoint_60_10_30_v1"
        if args.dataset_profile == "lora25"
        else "stratified_random_60_10_30_v1"
    )
    checkpoint_metadata = {
        "dataset_path": os.path.abspath(args.dataset_path),
        "selected_rx_list": list(selected_rx_list),
        "known_tx": list(range(args.initial_known_classes)),
        "training_day_index": 0,
        "development_ratio": float(args.development_ratio),
        "dataset_profile": args.dataset_profile,
        "sample_split_protocol": split_protocol_name,
        "closedset_backbone": args.closedset_backbone,
        "seed": int(args.seed),
        "use_supcon": bool(args.use_supcon),
        "supcon_weight": float(args.supcon_weight),
        "supcon_temperature": float(args.supcon_temperature),
        "training_recipe_version": TRAINING_RECIPE_VERSION,
        "closedset_validation_fraction": float(args.closedset_val_ratio),
        "rf_augmentation": bool(not args.disable_rf_augmentation),
        "supcon_projection_dim": int(args.projection_dim) if args.use_supcon else 0,
        "supcon_projection_hidden_dim": int(args.projection_hidden_dim) if args.use_supcon else 0,
    }
    if args.train_closedset or not os.path.exists(args.checkpoint):
        model = train_closedset_model(
            train_set,
            args.initial_known_classes,
            args.feat_dim,
            args.epochs,
            args.batch_size,
            args.lr,
            device,
            args.checkpoint,
            use_supcon=args.use_supcon,
            supcon_weight=args.supcon_weight,
            supcon_temperature=args.supcon_temperature,
            checkpoint_metadata=checkpoint_metadata,
            seed=args.seed,
            validation_fraction=args.closedset_val_ratio,
            use_rf_augmentation=not args.disable_rf_augmentation,
            projection_hidden_dim=args.projection_hidden_dim,
            projection_dim=args.projection_dim,
            validation_set=validation_set,
            model_factory=closedset_model_factory,
        )
    else:
        model = load_closedset_model(
            args.checkpoint,
            args.initial_known_classes,
            args.feat_dim,
            device,
            expected_metadata=checkpoint_metadata,
            model_factory=closedset_model_factory,
        )

    print("\n[Extract] Deep embeddings")
    Z_train, y_train = extract_deep_features(model, X_train, y_train, args.test_batch_size, device)
    # discovery 特征适配只允许访问 Day1 已知训练特征和当前轮 discovery 特征；
    # held-out eval embedding 在 discovery_only 分支之后才会抽取。
    args.discovery_adapter_known_Z = Z_train
    args.discovery_adapter_known_y = y_train
    if validation_set is None:
        day1_training_subset, day1_validation_subset = stratified_train_validation_split(
            train_set, validation_fraction=args.closedset_val_ratio, seed=args.seed
        )
        day1_validation_indices = np.asarray(day1_validation_subset.indices, dtype=np.int64)
        X_calibration = X_train[day1_validation_indices]
        y_calibration = y_train[day1_validation_indices]
        Z_calibration = Z_train[day1_validation_indices]
    else:
        # LoRa 的 IQ_7 是独立固定验证 transmission。深度特征和经典特征
        # 校准都只读取该 split，不从 IQ_1-6 训练集重新抽样。
        day1_training_subset = train_set
        day1_validation_subset = validation_set
        Z_calibration, y_calibration = extract_deep_features(
            model, X_validation, y_validation, args.test_batch_size, device
        )
        X_calibration = X_validation
    with open(os.path.join(args.save_dir, "split_protocol.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dataset_profile": args.dataset_profile,
            "split_level": (
                "transmission-disjoint fixed split"
                if args.dataset_profile == "lora25"
                else "stored-order block split (capture metadata unavailable)"
            ),
            "seed": int(args.seed),
            "day1_backbone_train_samples": int(len(day1_training_subset)),
            "day1_validation_samples": int(len(day1_validation_subset)),
            "day1_heldout_evaluation_samples": int(len(eval_data["eval_initial"]["y"])),
            "development_ratio": float(args.development_ratio),
            "validation_split_key": validation_split_key,
            "lora_recording_level_eval_enabled": bool(args.enable_lora_recording_eval),
            "lora_ssl_adaptation_enabled": bool(args.lora_ssl_adaptation),
            "lora_recording_ssl_adaptation_enabled": bool(args.lora_recording_ssl_adaptation),
            "frozen_before_unknown_rounds": [
                "checkpoint", "CF-LCG", "MV-ACC parameters",
                "grouped DOI candidate set and Day1 validation-only selection rule",
                "iCaRL fallback candidate set and Day1 validation-only selection rule",
            ],
        }, f, indent=2)
    args.cflcg_extractor = None
    args.cflcg_gate_open = True
    if not args.disable_cflcg:
        cflcg_selection, cflcg_extractor = select_classic_feature_view(
            X_calibration, y_calibration, args.cflcg_gate_threshold)
        args.cflcg_extractor = cflcg_extractor
        args.cflcg_gate_open = bool(cflcg_selection["rf_gate_open"])
        save_selection(cflcg_selection, args.save_dir)
        print(f"[CF-LCG] view={cflcg_selection['output_feature_group']} | purity={cflcg_selection['best_purity']:.3f} | gate_open={cflcg_selection['rf_gate_open']}")
        if not cflcg_selection["rf_gate_open"]:
            print("[CF-LCG] global binary gate closed; Local CF-LCG remains sample-adaptive.")
    if not args.disable_mvacc_calibration:
        calibrate_mvacc_on_known_day1(
            X_calibration, y_calibration, Z_calibration, args,
        )
    for rd in round_data:
        rd["Z"], rd["y"] = extract_deep_features(model, rd["X"], rd["y"], args.test_batch_size, device)

    if args.discovery_only:
        # 前端质量验证到这里就可以停止：只使用 Day1 train/validation 与当前轮
        # discovery 特征，不抽取 held-out eval embedding，也不进入 RADCIL/DOI 后端。
        discovery_rows = []
        print("\n[discovery_only] Strict GPCC/MV-ACC front-end evaluation")
        for i, rd in enumerate(round_data, start=1):
            row, _ = run_discovery(
                "MV-ACC",
                f"R{i}",
                f"Day {i + 1}",
                args.round_size,
                rd["X"],
                rd["y"],
                rd["Z"],
                args,
                recording_ids=rd.get("recording_id"),
            )
            row["Method"] = "MV-ACC-CIL discovery"
            discovery_rows.append(row)
        save_csv(discovery_rows, os.path.join(args.save_dir, "clustering_results.csv"))
        print("[discovery_only] Saved strict discovery outputs; skipped held-out eval feature extraction and CIL training.")
        print(f"Saved: {os.path.join(args.save_dir, 'clustering_results.csv')}")
        return

    eval_Z_dict = {}
    eval_X_dict = {}
    eval_y_dict = {}
    for key, ed in eval_data.items():
        eval_Z_dict[key], eval_y_dict[key] = extract_deep_features(model, ed["X"], ed["y"], args.test_batch_size, device)
        eval_X_dict[key] = ed["X"]

    # Baseline feature bank is built from the initial closed-set backbone only.
    # 这样 CIL baseline 使用同一套冻结表征与同一套 strict split，不会被后续
    # MV-ACC-CIL student 的逐轮更新影响；评估集特征只用于最终评估，不参与
    # 聚类、阈值校准或训练统计量拟合。
    proto_bank = build_proto_feature_bank(
        Z_train,
        X_train,
        [rd["Z"] for rd in round_data],
        [rd["X"] for rd in round_data],
        eval_Z_dict,
        eval_X_dict,
        args,
        cflcg_mode="MV-ACC",
    )
    graph_fusion_round_infos = []
    deep_only_round_infos = []
    if not args.disable_cil_baselines:
        # Deep-HDBSCAN 端到端 baseline 使用同一初始 backbone 的 deep embedding，
        # 并按每轮未知数据独立发现新类；这里不读取 held-out eval 真值。
        for i, rd in enumerate(round_data, start=1):
            _, deep_info = run_discovery("Deep only", f"R{i}", f"Day {i + 1}", args.round_size, rd["X"], rd["y"], rd["Z"], args)
            deep_only_round_infos.append(deep_info)

    # MV-ACC-CIL 使用冻结 Teacher 完成发现，再更新 Student。可选后端只改变
    # 评估时的预测后处理，不读取 held-out 真值，也不改变发现或训练流程。
    hybrid_doi_enabled = bool(args.radcil_grouped_doi_fusion) or float(args.radcil_doi_fusion_weight) > 0
    old_prototype_route_enabled = bool(args.radcil_old_prototype_route)
    prototype_memory_enabled = hybrid_doi_enabled or old_prototype_route_enabled
    icarl_fallback_enabled = bool(args.radcil_icarl_fallback)
    old_logit_bias_candidates = [
        float(value) for value in str(args.radcil_old_logit_bias_candidates).split(",")
        if value.strip()
    ]
    old_logit_bias_enabled = bool(old_logit_bias_candidates)
    if sum([bool(hybrid_doi_enabled), bool(old_prototype_route_enabled), bool(icarl_fallback_enabled), bool(old_logit_bias_enabled)]) > 1:
        raise ValueError("DOI-memory fusion, old-prototype routing, iCaRL fallback and old-logit bias must be tested as separate ablations.")
    main_method_name = (
        "MV-ACC-CIL + old-prototype-route"
        if old_prototype_route_enabled
        else (
            "MV-ACC-CIL + grouped DOI-memory"
            if args.radcil_grouped_doi_fusion
            else (
                "MV-ACC-CIL + iCaRL-fallback"
                if icarl_fallback_enabled
                else (
                    "MV-ACC-CIL + old-logit-bias"
                    if old_logit_bias_enabled
                    else ("MV-ACC-CIL + DOI-memory" if hybrid_doi_enabled else "MV-ACC-CIL")
                )
            )
        )
    )
    print(f"\n[{main_method_name}] End-to-end pseudo-label class-incremental learning")
    if hybrid_doi_enabled:
        print(
            "[Hybrid DOI] "
            f"fusion_weight={args.radcil_doi_fusion_weight}, "
            f"align_lambda={args.radcil_doi_align_lambda}, "
            f"prototype_temperature={args.radcil_doi_prototype_temperature}"
        )
    if old_prototype_route_enabled:
        print(
            "[Old prototype route] "
            f"candidates={args.radcil_old_prototype_route_candidates}, "
            f"align_lambda={args.radcil_doi_align_lambda}"
        )
    clustering_rows, incremental_rows, recording_level_rows, retention_rows = [], [], [], []
    joint_refinement_rows = []
    validation_retention_rows = []
    student = copy.deepcopy(model).to(device).eval()
    pseudo_to_true = {int(c): int(c) for c in range(args.initial_known_classes)}
    memory_x, memory_y = _update_iq_memory(None, None, X_train, y_train, args.memory_per_class, args.seed)
    doi_prototype_bank = None
    icarl_prototype_bank = None
    doi_fusion_weight = float(args.radcil_doi_fusion_weight)
    old_prototype_route_threshold = 1.0
    icarl_fallback_threshold = float(args.radcil_icarl_fallback_threshold)
    old_logit_bias = 0.0
    doi_calibration_rows = []
    old_prototype_route_calibration_rows = []
    icarl_fallback_calibration_rows = []
    old_logit_bias_calibration_rows = []
    if prototype_memory_enabled:
        doi_prototype_bank = _build_aligned_doi_prototype_bank(
            student,
            memory_x,
            memory_y,
            args.test_batch_size,
            device,
            previous_bank=None,
            align_lambda=args.radcil_doi_align_lambda,
        )
        if args.radcil_grouped_doi_fusion:
            candidates = [
                float(value) for value in args.radcil_doi_fusion_candidates.split(",")
                if value.strip()
            ]
            doi_fusion_weight, calibration = _calibrate_grouped_doi_weight_on_validation(
                student, X_calibration, y_calibration, doi_prototype_bank,
                args.test_batch_size, device, candidates,
                args.radcil_doi_prototype_temperature, args.initial_known_classes,
            )
            doi_calibration_rows.extend([{"Stage": "Initial", **row} for row in calibration])
            print(f"[Grouped DOI calibration] stage=Initial | Day1 validation only | weight={doi_fusion_weight}")
            initial_pred = _predict_hybrid_radcil_doi(
                student,
                eval_data["eval_initial"]["X"],
                doi_prototype_bank,
                args.test_batch_size,
                device,
                doi_fusion_weight,
                args.radcil_doi_prototype_temperature,
                old_class_count=args.initial_known_classes,
            )
        elif old_prototype_route_enabled:
            route_candidates = [
                float(value) for value in args.radcil_old_prototype_route_candidates.split(",")
                if value.strip()
            ]
            old_prototype_route_threshold, calibration = _calibrate_old_prototype_route_on_validation(
                student, X_calibration, y_calibration, doi_prototype_bank,
                args.test_batch_size, device, route_candidates, args.initial_known_classes,
            )
            old_prototype_route_calibration_rows.extend([{"Stage": "Initial", **row} for row in calibration])
            print(
                "[Old prototype route calibration] stage=Initial | Day1 validation only | "
                f"threshold={old_prototype_route_threshold}"
            )
            initial_pred = _predict_radcil_old_prototype_route(
                student,
                eval_data["eval_initial"]["X"],
                doi_prototype_bank,
                args.test_batch_size,
                device,
                old_mass_threshold=old_prototype_route_threshold,
                old_class_count=args.initial_known_classes,
            )
        else:
            initial_pred = _predict_hybrid_radcil_doi(
                student,
                eval_data["eval_initial"]["X"],
                doi_prototype_bank,
                args.test_batch_size,
                device,
                doi_fusion_weight,
                args.radcil_doi_prototype_temperature,
                old_class_count=None,
            )
    elif icarl_fallback_enabled:
        icarl_prototype_bank = _build_replay_prototype_bank(
            student, memory_x, memory_y, args.test_batch_size, device
        )
        candidates = [
            float(value) for value in args.radcil_icarl_fallback_candidates.split(",")
            if value.strip()
        ]
        icarl_fallback_threshold, calibration = _calibrate_icarl_fallback_threshold_on_validation(
            student, X_calibration, y_calibration, icarl_prototype_bank,
            args.test_batch_size, device, candidates, args.initial_known_classes,
        )
        icarl_fallback_calibration_rows.extend([{"Stage": "Initial", **row} for row in calibration])
        print(
            "[iCaRL fallback calibration] stage=Initial | Day1 validation only | "
            f"threshold={icarl_fallback_threshold}"
        )
        initial_pred = _predict_radcil_icarl_fallback(
            student,
            eval_data["eval_initial"]["X"],
            icarl_prototype_bank,
            args.test_batch_size,
            device,
            confidence_threshold=icarl_fallback_threshold,
            old_class_count=args.initial_known_classes,
        )
    elif old_logit_bias_enabled:
        old_logit_bias, calibration = _calibrate_old_logit_bias_on_validation(
            student, X_calibration, y_calibration, args.test_batch_size, device,
            old_logit_bias_candidates, args.initial_known_classes,
        )
        old_logit_bias_calibration_rows.extend([{"Stage": "Initial", **row} for row in calibration])
        print(
            "[Old-logit bias calibration] stage=Initial | Day1 validation only | "
            f"old_class_count={args.initial_known_classes} | bias={old_logit_bias}"
        )
        initial_pred = _predict_end_to_end(
            student, eval_data["eval_initial"]["X"], args.test_batch_size, device,
            old_class_count=args.initial_known_classes,
            old_logit_bias=old_logit_bias,
        )
    else:
        initial_pred = _predict_end_to_end(student, eval_data["eval_initial"]["X"], args.test_batch_size, device)
    initial_row = _evaluate_raw_predictions(
        main_method_name, "Initial", eval_data["eval_initial"]["day"], args.initial_known_classes,
        "day1_eval_initial_30pct", eval_y_dict["eval_initial"], initial_pred, pseudo_to_true,
        "-", "-", "-", args.initial_known_classes, args.round_size, None,
    )
    incremental_rows.append(initial_row)
    if args.save_end_to_end_eval_dumps:
        _save_end_to_end_eval_dump(
            args.save_dir,
            "Initial",
            eval_data["eval_initial"]["X"],
            eval_y_dict["eval_initial"],
            initial_pred,
            pseudo_to_true,
            args.test_batch_size,
            device,
            student,
            recording_id=eval_data["eval_initial"].get("recording_id"),
        )
    if args.enable_lora_recording_eval:
        recording_initial_row = _evaluate_recording_level_predictions(
            main_method_name,
            "Initial",
            eval_data["eval_initial"]["day"],
            args.initial_known_classes,
            "day1_eval_initial_recording_majority",
            eval_y_dict["eval_initial"],
            initial_pred,
            pseudo_to_true,
            eval_data["eval_initial"].get("recording_id"),
            "-",
            "-",
            "-",
            args.initial_known_classes,
            args.round_size,
            None,
        )
        if recording_initial_row is not None:
            recording_level_rows.append(recording_initial_row)
    retention_rows.append({"Round": "Initial", "Fixed Day1 Old-Class Acc": float(initial_row["Overall Acc"])})
    if X_validation is not None:
        if old_prototype_route_enabled:
            initial_validation_pred = _predict_radcil_old_prototype_route(
                student, X_validation, doi_prototype_bank,
                args.test_batch_size, device,
                old_mass_threshold=old_prototype_route_threshold,
                old_class_count=args.initial_known_classes,
            )
        else:
            initial_validation_pred = _predict_end_to_end(
                student, X_validation, args.test_batch_size, device,
                old_class_count=args.initial_known_classes if old_logit_bias_enabled else 0,
                old_logit_bias=old_logit_bias if old_logit_bias_enabled else 0.0,
            )
        validation_retention_rows.append({
            "Round": "Initial",
            "IQ_7 Validation Old-Class Acc": float(np.mean(initial_validation_pred == y_validation)),
        })
    initial_reference_acc = float(initial_row["Initial Known Acc"])
    recording_initial_reference_acc = (
        float(recording_level_rows[-1]["Initial Known Acc"]) if recording_level_rows else None
    )
    next_label = int(args.initial_known_classes)

    for i, rd in enumerate(round_data, start=1):
        teacher = copy.deepcopy(student).to(device).eval()
        lora_old_day_calibration_x = None
        lora_old_day_calibration_y = None
        if args.radcil_bn_recalibration:
            bn_recalibration_x = np.concatenate([memory_x, rd["X"]], axis=0)
            _recalibrate_batchnorm(
                teacher, bn_recalibration_x, args.test_batch_size, device,
                passes=args.radcil_bn_recalibration_passes,
                reset_running_stats=args.radcil_bn_recalibration_reset,
            )
            print(
                f"[BN recalibration] stage=Before R{i} discovery | "
                f"samples={len(bn_recalibration_x)} | reset={args.radcil_bn_recalibration_reset}"
            )
        if args.cross_day_repr_adaptation:
            # 适配发生在当前轮 discovery 之前。未知真实标签只在后续离线
            # clustering metrics 中使用，不进入适配损失、伪标签或模型选择。
            adaptation_result = adapt_model_cross_day(
                teacher,
                X_train,
                y_train,
                rd["X"],
                device=device,
                epochs=args.cross_day_repr_epochs,
                batch_size=args.cross_day_repr_batch_size,
                lr=args.cross_day_repr_lr,
                consistency_weight=args.cross_day_repr_consistency_weight,
                coral_weight=args.cross_day_repr_coral_weight,
                seed=args.seed + i,
            )
            teacher = adaptation_result.model
            rd["cross_day_adaptation_diagnostics"] = adaptation_result.diagnostics
        if args.lora_old_day_calibration_adaptation or args.lora_old_day_joint_cil:
            # 旧类跨天监督适配只读取当前日期已知旧设备的 IQ_1-7。
            # 当前轮未知设备和 IQ_8-10 held-out eval 均不进入训练。
            old_class_count = int(args.initial_known_classes + (i - 1) * args.round_size)
            calibration_transmissions = [
                int(item.strip())
                for item in str(args.lora_old_day_calibration_transmissions).split(",")
                if item.strip()
            ]
            calibration_x, calibration_y, calibration_rows = load_lora_old_day_calibration(
                args.lora_old_day_calibration_raw_dir,
                day=i + 1,
                old_class_count=old_class_count,
                transmissions=calibration_transmissions,
                symbols_per_transmission=args.lora_old_day_calibration_symbols,
                decimation=args.lora_old_day_calibration_decimation,
                representation=args.lora_old_day_calibration_representation,
            )
            lora_old_day_calibration_x = calibration_x
            lora_old_day_calibration_y = calibration_y
            if args.lora_old_day_calibration_adaptation:
                adaptation_result = adapt_model_lora_old_day_supervised(
                    teacher,
                    X_train,
                    y_train,
                    calibration_x,
                    calibration_y,
                    rd["X"],
                    device=device,
                    epochs=args.lora_old_day_calibration_epochs,
                    batch_size=args.lora_old_day_calibration_batch_size,
                    lr=args.lora_old_day_calibration_lr,
                    calibration_ce_weight=args.lora_old_day_calibration_ce_weight,
                    known_ce_weight=args.lora_old_day_calibration_known_ce_weight,
                    feature_alignment_weight=args.lora_old_day_calibration_feature_weight,
                    discovery_anchor_weight=args.lora_old_day_calibration_anchor_weight,
                    seed=args.seed + 100 + i,
                )
                teacher = adaptation_result.model
                rd["lora_old_day_calibration_diagnostics"] = {
                    **adaptation_result.diagnostics,
                    "Calibration Day": int(i + 1),
                    "Calibration Transmissions": calibration_transmissions,
                    "Calibration Manifest Rows": calibration_rows,
                }
            else:
                rd["lora_old_day_calibration_diagnostics"] = {
                    "Cross-Day Adapter": "joint_cil_only",
                    "Calibration Day": int(i + 1),
                    "Calibration Transmissions": calibration_transmissions,
                    "Calibration Manifest Rows": calibration_rows,
                    "Calibration Samples": int(len(calibration_x)),
                }
        if args.lora_ssl_adaptation:
            if args.dataset_profile != "lora25":
                raise ValueError("--lora_ssl_adaptation is only supported for the LoRa25 strict profile.")
            # LoRa SSL 适配只使用 Day1 已知训练标签和当前轮 discovery 无标签
            # 样本。它发生在聚类前端之前，不读取未知真值或 held-out eval。
            ssl_result = adapt_model_lora_ssl(
                teacher,
                X_train,
                y_train,
                rd["X"],
                device=device,
                epochs=args.lora_ssl_epochs,
                batch_size=args.lora_ssl_batch_size,
                lr=args.lora_ssl_lr,
                ce_weight=args.lora_ssl_ce_weight,
                instance_weight=args.lora_ssl_instance_weight,
                teacher_weight=args.lora_ssl_teacher_weight,
                temperature=args.lora_ssl_temperature,
                projection_hidden_dim=args.projection_hidden_dim,
                projection_dim=args.projection_dim,
                seed=args.seed + 100 + i,
            )
            teacher = ssl_result.model
            rd["lora_ssl_adaptation_diagnostics"] = ssl_result.diagnostics
        if args.lora_recording_ssl_adaptation:
            if args.dataset_profile != "lora25":
                raise ValueError("--lora_recording_ssl_adaptation is only supported for the LoRa25 strict profile.")
            if rd.get("recording_id") is None:
                raise ValueError("--lora_recording_ssl_adaptation requires LoRa recording_id metadata.")
            # Recording-level SSL 只把可观测 recording_id 当作 must-link 约束。
            # 它不知道当前轮真实设备标签，也不访问 IQ_8-10 held-out eval。
            recording_ssl_result = adapt_model_lora_recording_ssl(
                teacher,
                X_train,
                y_train,
                rd["X"],
                rd["recording_id"],
                device=device,
                epochs=args.lora_recording_ssl_epochs,
                batch_size=args.lora_recording_ssl_batch_size,
                lr=args.lora_recording_ssl_lr,
                ce_weight=args.lora_recording_ssl_ce_weight,
                recording_weight=args.lora_recording_ssl_recording_weight,
                instance_weight=args.lora_recording_ssl_instance_weight,
                teacher_weight=args.lora_recording_ssl_teacher_weight,
                temperature=args.lora_recording_ssl_temperature,
                projection_hidden_dim=args.projection_hidden_dim,
                projection_dim=args.projection_dim,
                seed=args.seed + 200 + i,
            )
            teacher = recording_ssl_result.model
            rd["lora_recording_ssl_adaptation_diagnostics"] = recording_ssl_result.diagnostics
        # Teacher is frozen during discovery. This prevents moving representations from changing clusters.
        rd["Z"], _ = extract_deep_features(teacher, rd["X"], rd["y"], args.test_batch_size, device)
        row, info = run_discovery(
            "MV-ACC",
            f"R{i}",
            f"Day {i + 1}",
            args.round_size,
            rd["X"],
            rd["y"],
            rd["Z"],
            args,
            recording_ids=rd.get("recording_id"),
        )
        row["Method"] = "MV-ACC-CIL discovery"
        if rd.get("cross_day_adaptation_diagnostics"):
            row.update(rd["cross_day_adaptation_diagnostics"])
        if rd.get("lora_ssl_adaptation_diagnostics"):
            row.update(rd["lora_ssl_adaptation_diagnostics"])
        if rd.get("lora_recording_ssl_adaptation_diagnostics"):
            row.update(rd["lora_recording_ssl_adaptation_diagnostics"])
        clustering_rows.append(row)
        graph_fusion_round_infos.append(info)
        if args.discovery_only:
            # LoRa/WiSig 发现前端验证只比较聚类标签和离线聚类指标；这里跳过
            # RADCIL/混合后端训练，保证 seed7 短任务聚焦在 GPCC 是否改善聚类。
            continue
        labels = np.asarray(info["labels"], dtype=np.int64)
        if np.any(labels < 0):
            raise RuntimeError("MV-ACC-CIL requires no-drop labels; found an unassigned noise sample.")
        cluster_ids = sorted(np.unique(labels).tolist())
        cid_to_pseudo = {int(cid): int(next_label + j) for j, cid in enumerate(cluster_ids)}
        pseudo_y = np.asarray([cid_to_pseudo[int(c)] for c in labels], dtype=np.int64)
        pairs = [(int(cid), int(cid_to_pseudo[int(cid)])) for cid in cluster_ids]
        pseudo_to_true.update(build_posthoc_pseudo_to_true(rd["y"], labels, pairs))
        raw_probs = np.asarray(
            info.get("raw_hdbscan_probabilities", np.ones(len(labels))),
            dtype=np.float32,
        )
        raw_noise = np.asarray(info.get("raw_noise_mask", np.zeros(len(labels), dtype=bool)), dtype=bool)
        current_w = _compute_pseudo_sample_weights(
            labels,
            raw_probs,
            reliability_map=info.get("reliability_map", info.get("reliability", {})),
            floor=args.pseudo_weight_cluster_reliability_floor
            if args.pseudo_weight_use_cluster_reliability
            else args.pseudo_weight_floor,
            use_cluster_reliability=args.pseudo_weight_use_cluster_reliability,
        )
        current_w[raw_noise] = float(args.pseudo_weight_floor)
        old_out = int(student.classifier.out_features)
        registration_mask = _select_pseudo_registration_mask(
            pseudo_y,
            current_w,
            top_fraction=float(args.radcil_registration_top_fraction),
            min_per_class=int(args.radcil_registration_min_per_class),
        )
        if not np.all(registration_mask):
            print(
                f"[joint registration] R{i}: kept={int(np.sum(registration_mask))}/{len(registration_mask)} "
                f"top_fraction={float(args.radcil_registration_top_fraction):.2f}; "
                "all discovery samples remain in current-round training"
            )
        imprinted_means = np.stack(
            [
                rd["Z"][
                    (pseudo_y == int(next_label + j)) & registration_mask
                ].mean(axis=0)
                for j in range(len(cluster_ids))
            ]
        ).astype(np.float32)
        next_label += len(cluster_ids)
        student = _expand_closedset_classifier(
            teacher,
            next_label,
            device,
            imprinted_means,
            new_imprint_scale=args.radcil_new_imprint_scale,
            new_imprint_bias=args.radcil_new_imprint_bias,
        )
        # 注册和训练故意分离：低置信样本仍参与加权 CE/SupCon，避免新类样本量
        # 因为注册筛选而塌缩；只有 replay memory 使用高置信注册子集。
        student = _train_end_to_end_cil(
            student,
            teacher,
            rd["X"],
            pseudo_y,
            current_w,
            memory_x,
            memory_y,
            old_out,
            args,
            device,
            current_recording_id=rd.get("recording_id"),
            old_day_calibration_x=lora_old_day_calibration_x,
            old_day_calibration_y=lora_old_day_calibration_y,
        )
        if int(args.radcil_balanced_head_recalibration_epochs) > 0:
            student = _recalibrate_balanced_classifier_head(
                student,
                rd["X"],
                pseudo_y,
                current_w,
                memory_x,
                memory_y,
                args.radcil_balanced_head_recalibration_epochs,
                args.incremental_batch_size,
                device,
                old_out_dim=old_out,
                new_distill_weight=args.radcil_balanced_head_new_distill_weight,
            )
        if args.enable_joint_discovery_refinement:
            # 第一轮 CIL 后，重新观察当前 Student 的类别结构。二次发现只读取
            # 当前轮 discovery 样本，随后用无标签中心匹配回原伪类编号。
            refined_z, _ = extract_deep_features(
                student,
                rd["X"],
                rd["y"],
                args.test_batch_size,
                device,
            )
            refined_row, refined_info = run_discovery(
                "MV-ACC",
                f"R{i}_refined",
                f"Day {i + 1}",
                args.round_size,
                rd["X"],
                rd["y"],
                refined_z,
                args,
                recording_ids=rd.get("recording_id"),
            )
            refined_row["Method"] = "Joint discovery refinement"
            joint_refinement_rows.append(refined_row)
            refined_pseudo_y = _align_refined_clusters_to_pseudo(
                rd["Z"],
                pseudo_y,
                refined_z,
                refined_info["labels"],
            )
            refined_filter_mask = np.ones(len(refined_pseudo_y), dtype=bool)
            if float(args.radcil_refined_filter_top_fraction) < 0.999:
                # Stage56：二次发现后不再盲目把所有对齐伪标签喂回第二次 CIL。
                # 这里用同一个 Student 的分类头 logits 做一致性检查，只保留
                # Student 预测与 refined 伪标签一致且类内置信度靠前的样本。
                # 整个过程只依赖当前轮 discovery 样本、伪标签和训练期模型输出，
                # 不读取未知真值或 held-out eval。
                _, refined_logits = _extract_end_to_end_outputs(
                    student,
                    rd["X"],
                    args.test_batch_size,
                    device,
                )
                refined_filter_mask, refined_filter_diag = _select_refined_consensus_training_mask(
                    refined_pseudo_y,
                    refined_logits,
                    top_fraction=float(args.radcil_refined_filter_top_fraction),
                    min_per_class=int(args.radcil_refined_filter_min_per_class),
                )
                refined_row.update(refined_filter_diag)
                print(
                    f"[refined filter] R{i}: kept={int(np.sum(refined_filter_mask))}/{len(refined_filter_mask)} "
                    f"top_fraction={float(args.radcil_refined_filter_top_fraction):.2f} "
                    f"agreement={float(refined_filter_diag['Refined Filter Agreement Rate']):.4f}"
                )
            else:
                refined_row.update({
                    "Refined Filter Kept": int(len(refined_pseudo_y)),
                    "Refined Filter Total": int(len(refined_pseudo_y)),
                    "Refined Filter Keep Rate": 1.0,
                    "Refined Filter Agreement Rate": 1.0,
                    "Refined Filter Mean Confidence": 1.0,
                    "Refined Filter Fallback Classes": 0,
                })
            refined_recording_id = rd.get("recording_id")
            refined_recording_id = (
                None
                if refined_recording_id is None
                else np.asarray(refined_recording_id, dtype=np.int64)[refined_filter_mask]
            )
            student = _train_end_to_end_cil(
                student,
                student,
                rd["X"][refined_filter_mask],
                refined_pseudo_y[refined_filter_mask],
                current_w[refined_filter_mask],
                memory_x,
                memory_y,
                old_out,
                args,
                device,
                current_recording_id=refined_recording_id,
                old_day_calibration_x=lora_old_day_calibration_x,
                old_day_calibration_y=lora_old_day_calibration_y,
            )
            if int(args.radcil_balanced_head_recalibration_epochs) > 0:
                student = _recalibrate_balanced_classifier_head(
                    student,
                    rd["X"][refined_filter_mask],
                    refined_pseudo_y[refined_filter_mask],
                    current_w[refined_filter_mask],
                    memory_x,
                    memory_y,
                    args.radcil_balanced_head_recalibration_epochs,
                    args.incremental_batch_size,
                    device,
                    old_out_dim=old_out,
                    new_distill_weight=args.radcil_balanced_head_new_distill_weight,
                )
        memory_labels = refined_pseudo_y if args.enable_joint_discovery_refinement else pseudo_y
        memory_mask = registration_mask
        if args.enable_joint_discovery_refinement and float(args.radcil_refined_filter_top_fraction) < 0.999:
            memory_mask = registration_mask & refined_filter_mask
            if not np.any(memory_mask):
                memory_mask = registration_mask
        memory_source_x = rd["X"][memory_mask]
        memory_source_y = memory_labels[memory_mask]
        memory_x, memory_y = _update_iq_memory(
            memory_x,
            memory_y,
            memory_source_x,
            memory_source_y,
            args.memory_per_class,
            args.seed + i,
        )
        if args.radcil_bn_recalibration:
            _recalibrate_batchnorm(
                student, memory_x, args.test_batch_size, device,
                passes=args.radcil_bn_recalibration_passes,
                reset_running_stats=args.radcil_bn_recalibration_reset,
            )
            print(
                f"[BN recalibration] stage=After R{i} training | "
                f"samples={len(memory_x)} | reset={args.radcil_bn_recalibration_reset}"
            )
        eval_key = f"eval_r{i}"
        if prototype_memory_enabled:
            doi_prototype_bank = _build_aligned_doi_prototype_bank(
                student,
                memory_x,
                memory_y,
                args.test_batch_size,
                device,
                previous_bank=doi_prototype_bank,
                align_lambda=args.radcil_doi_align_lambda,
            )
            if args.radcil_grouped_doi_fusion:
                doi_fusion_weight, calibration = _calibrate_grouped_doi_weight_on_validation(
                    student, X_calibration, y_calibration, doi_prototype_bank,
                    args.test_batch_size, device, candidates,
                    args.radcil_doi_prototype_temperature, old_out,
                )
                doi_calibration_rows.extend([{"Stage": f"After R{i}", **row} for row in calibration])
                print(
                    f"[Grouped DOI calibration] stage=After R{i} | Day1 validation only | "
                    f"old_class_count={old_out} | weight={doi_fusion_weight}"
                )
            if old_prototype_route_enabled:
                route_candidates = [
                    float(value) for value in args.radcil_old_prototype_route_candidates.split(",")
                    if value.strip()
                ]
                old_prototype_route_threshold, calibration = _calibrate_old_prototype_route_on_validation(
                    student, X_calibration, y_calibration, doi_prototype_bank,
                    args.test_batch_size, device, route_candidates, old_out,
                )
                old_prototype_route_calibration_rows.extend([{"Stage": f"After R{i}", **row} for row in calibration])
                print(
                    f"[Old prototype route calibration] stage=After R{i} | Day1 validation only | "
                    f"threshold={old_prototype_route_threshold}"
                )
                pred = _predict_radcil_old_prototype_route(
                    student,
                    eval_data[eval_key]["X"],
                    doi_prototype_bank,
                    args.test_batch_size,
                    device,
                    old_mass_threshold=old_prototype_route_threshold,
                    old_class_count=old_out,
                )
            else:
                pred = _predict_hybrid_radcil_doi(
                    student,
                    eval_data[eval_key]["X"],
                    doi_prototype_bank,
                    args.test_batch_size,
                    device,
                    doi_fusion_weight,
                    args.radcil_doi_prototype_temperature,
                    old_class_count=old_out if args.radcil_grouped_doi_fusion else None,
                )
        elif icarl_fallback_enabled:
            icarl_prototype_bank = _build_replay_prototype_bank(
                student, memory_x, memory_y, args.test_batch_size, device
            )
            icarl_fallback_threshold, calibration = _calibrate_icarl_fallback_threshold_on_validation(
                student, X_calibration, y_calibration, icarl_prototype_bank,
                args.test_batch_size, device, candidates, old_out,
            )
            icarl_fallback_calibration_rows.extend([{"Stage": f"After R{i}", **row} for row in calibration])
            print(
                f"[iCaRL fallback calibration] stage=After R{i} | Day1 validation only | "
                f"old_class_count={old_out} | threshold={icarl_fallback_threshold}"
            )
            pred = _predict_radcil_icarl_fallback(
                student,
                eval_data[eval_key]["X"],
                icarl_prototype_bank,
                args.test_batch_size,
                device,
                confidence_threshold=icarl_fallback_threshold,
                old_class_count=old_out,
            )
        elif old_logit_bias_enabled:
            old_logit_bias, calibration = _calibrate_old_logit_bias_on_validation(
                student, X_calibration, y_calibration, args.test_batch_size, device,
                old_logit_bias_candidates, old_out,
            )
            old_logit_bias_calibration_rows.extend([{"Stage": f"After R{i}", **row} for row in calibration])
            print(
                f"[Old-logit bias calibration] stage=After R{i} | Day1 validation only | "
                f"old_class_count={old_out} | bias={old_logit_bias}"
            )
            pred = _predict_end_to_end(
                student, eval_data[eval_key]["X"], args.test_batch_size, device,
                old_class_count=old_out,
                old_logit_bias=old_logit_bias,
            )
        else:
            pred = _predict_end_to_end(student, eval_data[eval_key]["X"], args.test_batch_size, device)
        incremental_rows.append(_evaluate_raw_predictions(
            main_method_name, f"After R{i}", eval_data[eval_key]["day"], args.initial_known_classes + i * args.round_size,
            f"{eval_key}_30pct_{args.initial_known_classes + i * args.round_size}_seen", eval_y_dict[eval_key], pred,
            pseudo_to_true, args.round_size, info["discovered_clusters"], len(cluster_ids),
            args.initial_known_classes, args.round_size, initial_reference_acc,
        ))
        if args.save_end_to_end_eval_dumps:
            _save_end_to_end_eval_dump(
                args.save_dir,
                f"After R{i}",
                eval_data[eval_key]["X"],
                eval_y_dict[eval_key],
                pred,
                pseudo_to_true,
                args.test_batch_size,
                device,
                student,
                recording_id=eval_data[eval_key].get("recording_id"),
            )
        if args.enable_lora_recording_eval:
            recording_row = _evaluate_recording_level_predictions(
                main_method_name,
                f"After R{i}",
                eval_data[eval_key]["day"],
                args.initial_known_classes + i * args.round_size,
                f"{eval_key}_recording_majority_{args.initial_known_classes + i * args.round_size}_seen",
                eval_y_dict[eval_key],
                pred,
                pseudo_to_true,
                eval_data[eval_key].get("recording_id"),
                args.round_size,
                info["discovered_clusters"],
                len(cluster_ids),
                args.initial_known_classes,
                args.round_size,
                recording_initial_reference_acc,
            )
            if recording_row is not None:
                recording_level_rows.append(recording_row)
        if prototype_memory_enabled:
            if old_prototype_route_enabled:
                fixed_pred = _predict_radcil_old_prototype_route(
                    student,
                    eval_data["eval_initial"]["X"],
                    doi_prototype_bank,
                    args.test_batch_size,
                    device,
                    old_mass_threshold=old_prototype_route_threshold,
                    old_class_count=old_out,
                )
            else:
                fixed_pred = _predict_hybrid_radcil_doi(
                    student,
                    eval_data["eval_initial"]["X"],
                    doi_prototype_bank,
                    args.test_batch_size,
                    device,
                    doi_fusion_weight,
                    args.radcil_doi_prototype_temperature,
                    old_class_count=old_out if args.radcil_grouped_doi_fusion else None,
                )
        elif icarl_fallback_enabled:
            fixed_pred = _predict_radcil_icarl_fallback(
                student,
                eval_data["eval_initial"]["X"],
                icarl_prototype_bank,
                args.test_batch_size,
                device,
                confidence_threshold=icarl_fallback_threshold,
                old_class_count=old_out,
            )
        elif old_logit_bias_enabled:
            fixed_pred = _predict_end_to_end(
                student, eval_data["eval_initial"]["X"], args.test_batch_size, device,
                old_class_count=old_out,
                old_logit_bias=old_logit_bias,
            )
        else:
            fixed_pred = _predict_end_to_end(student, eval_data["eval_initial"]["X"], args.test_batch_size, device)
        fixed_true = eval_y_dict["eval_initial"]
        fixed_mapped = np.asarray([pseudo_to_true.get(int(p), int(p)) for p in fixed_pred], dtype=np.int64)
        retention_rows.append({"Round": f"R{i}", "Fixed Day1 Old-Class Acc": float(np.mean(fixed_true == fixed_mapped))})
        if X_validation is not None:
            # 该表专供 seed7 训练配置选择。只评估固定 IQ_7 初始旧类，不能
            # 表示增量新类质量；IQ_8-10 held-out 指标不得用于候选选择。
            if old_prototype_route_enabled:
                validation_pred = _predict_radcil_old_prototype_route(
                    student, X_validation, doi_prototype_bank,
                    args.test_batch_size, device,
                    old_mass_threshold=old_prototype_route_threshold,
                    old_class_count=old_out,
                )
            else:
                validation_pred = _predict_end_to_end(
                    student, X_validation, args.test_batch_size, device,
                    old_class_count=old_out if old_logit_bias_enabled else 0,
                    old_logit_bias=old_logit_bias if old_logit_bias_enabled else 0.0,
                )
            validation_retention_rows.append({
                "Round": f"R{i}",
                "IQ_7 Validation Old-Class Acc": float(np.mean(validation_pred == y_validation)),
            })
        torch.save({
            "model_state": student.state_dict(),
            "round": i,
            "num_outputs": int(student.classifier.out_features),
            "method": main_method_name,
            "radcil_doi_fusion_weight": float(args.radcil_doi_fusion_weight),
            "radcil_doi_effective_fusion_weight": float(doi_fusion_weight),
            "radcil_grouped_doi_fusion": bool(args.radcil_grouped_doi_fusion),
            "radcil_prototype_anchor_weight": float(args.radcil_prototype_anchor_weight),
            "radcil_old_replay_supcon_weight": float(args.radcil_old_replay_supcon_weight),
            "radcil_aug_consistency_weight": float(args.radcil_aug_consistency_weight),
            "radcil_domain_alignment_weight": float(args.radcil_domain_alignment_weight),
            "radcil_metric_weight": float(args.radcil_metric_weight),
            "radcil_metric_scale": float(args.radcil_metric_scale),
            "radcil_metric_margin": float(args.radcil_metric_margin),
            "pseudo_weight_use_cluster_reliability": bool(args.pseudo_weight_use_cluster_reliability),
            "pseudo_weight_cluster_reliability_floor": float(
                args.pseudo_weight_cluster_reliability_floor
            ),
            "radcil_doi_align_lambda": float(args.radcil_doi_align_lambda),
            "radcil_doi_prototype_temperature": float(args.radcil_doi_prototype_temperature),
            "radcil_old_prototype_route": bool(old_prototype_route_enabled),
            "radcil_old_prototype_route_threshold": float(old_prototype_route_threshold),
            "radcil_icarl_fallback": bool(icarl_fallback_enabled),
            "radcil_icarl_fallback_threshold": float(icarl_fallback_threshold),
            "radcil_old_logit_bias": float(old_logit_bias),
        }, os.path.join(args.save_dir, f"mvacc_cil_after_r{i}.pth"))
        np.savez_compressed(os.path.join(args.save_dir, f"replay_memory_after_r{i}.npz"), X=memory_x, y=memory_y)
        if prototype_memory_enabled:
            np.savez_compressed(
                os.path.join(args.save_dir, f"hybrid_doi_prototype_bank_after_r{i}.npz"),
                prototypes=doi_prototype_bank[0],
                labels=doi_prototype_bank[1],
            )
        if icarl_fallback_enabled:
            np.savez_compressed(
                os.path.join(args.save_dir, f"icarl_fallback_prototype_bank_after_r{i}.npz"),
                prototypes=icarl_prototype_bank[0],
                labels=icarl_prototype_bank[1],
            )

    clustering_df = save_csv(clustering_rows, os.path.join(args.save_dir, "clustering_results.csv"))
    if args.enable_joint_discovery_refinement:
        save_csv(
            joint_refinement_rows,
            os.path.join(args.save_dir, "joint_discovery_refinement_results.csv"),
        )
    if args.discovery_only:
        print("[discovery_only] Saved strict discovery outputs; skipped CIL training and evaluation.")
        print(f"Saved: {os.path.join(args.save_dir, 'clustering_results.csv')}")
        return
    incremental_df = save_csv(incremental_rows, os.path.join(args.save_dir, "incremental_results.csv"))
    if recording_level_rows:
        save_csv(
            recording_level_rows,
            os.path.join(args.save_dir, "recording_level_incremental_results.csv"),
        )
    save_csv(retention_rows, os.path.join(args.save_dir, "fixed_day1_retention.csv"))
    if validation_retention_rows:
        save_csv(
            validation_retention_rows,
            os.path.join(args.save_dir, "iq7_validation_retention.csv"),
        )
    if doi_calibration_rows:
        save_csv(doi_calibration_rows, os.path.join(args.save_dir, "grouped_doi_iq7_calibration.csv"))
    if old_prototype_route_calibration_rows:
        save_csv(
            old_prototype_route_calibration_rows,
            os.path.join(args.save_dir, "old_prototype_route_iq7_calibration.csv"),
        )
    if icarl_fallback_calibration_rows:
        save_csv(
            icarl_fallback_calibration_rows,
            os.path.join(args.save_dir, "icarl_fallback_iq7_calibration.csv"),
        )
    if old_logit_bias_calibration_rows:
        save_csv(
            old_logit_bias_calibration_rows,
            os.path.join(args.save_dir, "old_logit_bias_iq7_calibration.csv"),
        )
    per_round_summary_df = build_per_round_summary(clustering_df, incremental_df)
    if not clustering_df.empty and "MV-ACC-CIL discovery" in set(clustering_df["Method"].astype(str)):
        c = clustering_df[clustering_df["Method"].astype(str) == "MV-ACC-CIL discovery"].set_index("Round")
        inc = incremental_df[incremental_df["Method"].astype(str) == main_method_name].copy()
        cil_rows = []
        for _, r in inc[inc["Stage"].astype(str).str.startswith("After R")].iterrows():
            rnd = str(r["Stage"]).replace("After ", "")
            d = c.loc[rnd]
            cil_rows.append({"Method":main_method_name, "Round":rnd, "Cluster Count":int(d["Final Cluster Count"]), "Incremental Pseudo-label Classes":int(r["Enrolled Clusters"]), "Cluster Count Error":int(d["Cluster Count Error"]), "Sample Coverage":float(d["Assignment Coverage"]), "Purity":float(d["Purity"]), "NMI":float(d["NMI"]), "ARI":float(d["ARI"]), "Hungarian Acc":float(d["Hungarian Acc"]), "Overall Acc":float(r["Overall Acc"]), "New Acc":float(r["New Acc"]), "Forgetting Rate":float(r["Forgetting Rate"])})
        per_round_summary_df = pd.DataFrame(cil_rows)
    save_csv(per_round_summary_df.to_dict("records"), os.path.join(args.save_dir, "per_round_summary_results.csv"))

    shared_baseline_df = pd.DataFrame()
    shared_comparison_df = pd.DataFrame()
    end_to_end_baseline_df = pd.DataFrame()
    end_to_end_comparison_df = pd.DataFrame()
    full_system_comparison_df = pd.DataFrame()

    if not args.disable_cil_baselines:
        # ------------------------------------------------------------
        # A) Shared-discovery controlled comparison
        # Every incremental learner receives the same high-quality pseudo-labels
        # produced by the proposed Graph-fusion discovery front-end. This isolates
        # the incremental retention / anti-forgetting component.
        # ------------------------------------------------------------
        if graph_fusion_round_infos is not None:
            shared_baseline_df = run_incremental_learning_baselines(
                graph_fusion_round_infos,
                proto_bank,
                y_train,
                round_data,
                eval_data,
                eval_y_dict,
                args,
                device,
                method_prefix="",
                comparison_label="Shared Graph-fusion pseudo-labels",
                feat_type="hybrid",
            )
            save_csv(shared_baseline_df.to_dict("records"), os.path.join(args.save_dir, "shared_discovery_baseline_results.csv"))
            shared_comparison_df = build_comparison_summary(
                shared_baseline_df,
                incremental_df,
                proposed_source_method=main_method_name,
                proposed_name=main_method_name,
            )
            save_csv(shared_comparison_df.to_dict("records"), os.path.join(args.save_dir, "shared_discovery_comparison_results.csv"))

            # Backward-compatible aliases for earlier scripts/results.
            save_csv(shared_baseline_df.to_dict("records"), os.path.join(args.save_dir, "incremental_baseline_results.csv"))
            save_csv(shared_comparison_df.to_dict("records"), os.path.join(args.save_dir, "sota_comparison_results.csv"))

        # ------------------------------------------------------------
        # B) End-to-end open-set incremental comparison
        # Conventional CIL learners are coupled with a neutral Deep-feature +
        # HDBSCAN discovery front-end, while Ours keeps its own adaptive multi-view
        # Graph-fusion discovery. Thus the table measures discovery quality and
        # incremental retention jointly.
        # ------------------------------------------------------------
        if deep_only_round_infos is not None:
            end_to_end_baseline_df = run_incremental_learning_baselines(
                deep_only_round_infos,
                proto_bank,
                y_train,
                round_data,
                eval_data,
                eval_y_dict,
                args,
                device,
                method_prefix="Deep-HDBSCAN + ",
                comparison_label="End-to-end with neutral Deep-HDBSCAN discovery",
                feat_type="deep",
            )
            save_csv(end_to_end_baseline_df.to_dict("records"), os.path.join(args.save_dir, "end_to_end_baseline_results.csv"))
            end_to_end_comparison_df = build_comparison_summary(
                end_to_end_baseline_df,
                incremental_df,
                proposed_source_method=main_method_name,
                proposed_name=main_method_name,
            )
            save_csv(end_to_end_comparison_df.to_dict("records"), os.path.join(args.save_dir, "end_to_end_comparison_results.csv"))

            # Paper-facing main table: discovery quality + incremental learning quality.
            full_system_comparison_df = build_full_end_to_end_system_comparison(
                end_to_end_baseline_df,
                incremental_df,
                proposed_source_method=main_method_name,
                proposed_display_name=main_method_name,
                proposed_incremental_module=(
                    "Network replay + old-prototype route"
                    if old_prototype_route_enabled
                    else (
                    "Network replay + confidence-gated exemplar fallback"
                    if icarl_fallback_enabled
                    else "Network replay + DOI-memory late fusion"
                    if hybrid_doi_enabled
                    else "Network replay + distillation"
                    )
                ),
            )
            save_csv(
                full_system_comparison_df.to_dict("records"),
                os.path.join(args.save_dir, "end_to_end_system_comparison.csv"),
            )

    if args.save_detailed_visualizations:
        plot_core_method_visualization_panels(args)

    pd.set_option("display.max_columns", 80)
    pd.set_option("display.width", 240)

    print("\n========== Table 1: Unknown Discovery / Clustering Results ==========")
    print(clustering_df.to_string(index=False))
    print(f"Saved: {os.path.join(args.save_dir, 'clustering_results.csv')}")

    print("\n========== Table 2: Incremental Recognition Results ==========")
    print(incremental_df.to_string(index=False))
    print(f"Saved: {os.path.join(args.save_dir, 'incremental_results.csv')}")

    print("\n========== Table 3: Per-Round Summary Results ==========")
    # This is the compact table for reporting: Method/Round + the six requested metrics only.
    print(per_round_summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved: {os.path.join(args.save_dir, 'per_round_summary_results.csv')}")

    if shared_baseline_df is not None and not shared_baseline_df.empty:
        print("\n========== Table 4: Shared-Discovery Baseline Details ==========")
        print(shared_baseline_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        print(f"Saved: {os.path.join(args.save_dir, 'shared_discovery_baseline_results.csv')}")

    if shared_comparison_df is not None and not shared_comparison_df.empty:
        print("\n========== Table 5: Shared-Discovery Controlled Comparison ==========")
        print(shared_comparison_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        print(f"Saved: {os.path.join(args.save_dir, 'shared_discovery_comparison_results.csv')}")

    if end_to_end_comparison_df is not None and not end_to_end_comparison_df.empty:
        print("\n========== Table 6: End-to-End Open-Set Incremental Comparison ==========")
        print(end_to_end_comparison_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        print(f"Saved: {os.path.join(args.save_dir, 'end_to_end_comparison_results.csv')}")

    if full_system_comparison_df is not None and not full_system_comparison_df.empty:
        print("\n========== Table 7: Full End-to-End System Comparison (Main Table) ==========")
        print(full_system_comparison_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        print(f"Saved: {os.path.join(args.save_dir, 'end_to_end_system_comparison.csv')}")

    if args.enable_visualization:
        print(f"Visualizations saved under: {os.path.join(args.save_dir, 'visualizations')}")
        print(f"Core comparison panels saved under: {os.path.join(args.save_dir, 'visualizations', 'core_comparison')}")


if __name__ == "__main__":
    main()
