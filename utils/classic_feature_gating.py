"""CF-LCG: calibration-only classical-feature selection for MV-ACC.

The selector sees only the Day-1 validation subset.  It never uses unknown
round labels, held-out evaluation samples, or the true number of new devices.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from features.rf_features import extract_rf_features_batch


def extended_hardware_features(x: np.ndarray) -> np.ndarray:
    """24-D base RF view plus 24 stable higher-order statistics (48-D)."""
    base = extract_rf_features_batch(x)
    z = np.asarray(x, dtype=np.float32)
    if z.ndim != 3:
        raise ValueError("Expected batched IQ samples with shape (N, 2, L) or (N, L, 2).")
    if z.shape[1] == 2:
        z = np.transpose(z, (0, 2, 1))  # Network layout (N, 2, L) -> (N, L, 2)
    elif z.shape[-1] != 2:
        raise ValueError("Expected batched IQ samples with shape (N, 2, L) or (N, L, 2).")
    iq = z[..., 0] + 1j * z[..., 1]
    amp = np.abs(iq)
    phase = np.unwrap(np.angle(iq), axis=1)
    freq = np.diff(phase, axis=1)
    stats = []
    for signal in (amp, phase, freq):
        mean = signal.mean(axis=1)
        std = signal.std(axis=1)
        centered = signal - mean[:, None]
        denom = std[:, None] + 1e-6
        skew = np.mean((centered / denom) ** 3, axis=1)
        kurt = np.mean((centered / denom) ** 4, axis=1)
        q25, q75 = np.quantile(signal, [0.25, 0.75], axis=1)
        peak = np.max(np.abs(signal), axis=1)
        energy = np.mean(signal ** 2, axis=1)
        stats.extend([mean, std, skew, kurt, q25, q75, peak, energy])
    return np.concatenate([base, np.stack(stats, axis=1).astype(np.float32)], axis=1).astype(np.float32)


def feature_groups(x: np.ndarray) -> Dict[str, np.ndarray]:
    base = extract_rf_features_batch(x)
    extended = extended_hardware_features(x)
    return {
        "amplitude_8d": base[:, :8],
        "phase_frequency_4d": base[:, 8:12],
        "iq_imbalance_9d": base[:, 12:21],
        "spectral_3d": base[:, 21:24],
        "all24_base": base,
        "all48_extended": extended,
    }


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float32))
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)


def knn_purity(x: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    """Label-only Day-1 calibration statistic; no test/unknown labels used."""
    x, y = _standardize(x), np.asarray(y)
    if len(y) < 3:
        return 0.0
    d = pairwise_distances(x, metric="euclidean")
    np.fill_diagonal(d, np.inf)
    neighbors = np.argsort(d, axis=1)[:, : min(k, len(y) - 1)]
    return float(np.mean(y[neighbors] == y[:, None]))


def local_cross_view_consistency(deep: np.ndarray, rf: np.ndarray, k: int = 12) -> np.ndarray:
    """Unsupervised per-sample agreement between deep and RF neighborhoods.

    Unknown labels are deliberately never used.  A sample receives high RF
    reliability only when its k-nearest neighbors agree across the two views.
    """
    deep, rf = _standardize(deep), _standardize(rf)
    n = len(deep)
    if n < 3:
        return np.ones(n, dtype=np.float32)
    k = min(max(2, int(k)), n - 1)
    nd = NearestNeighbors(n_neighbors=k + 1).fit(deep).kneighbors(return_distance=False)[:, 1:]
    nr = NearestNeighbors(n_neighbors=k + 1).fit(rf).kneighbors(return_distance=False)[:, 1:]
    score = np.empty(n, dtype=np.float32)
    for i in range(n):
        overlap = len(set(nd[i]).intersection(nr[i])) / float(k)
        score[i] = 0.10 + 0.90 * overlap  # avoid a numerically zero RF graph
    return score


def select_classic_feature_view(
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    gate_threshold: float = 0.80,
    k: int = 10,
) -> Tuple[dict, callable]:
    """Select the best classical view or explicitly close the RF gate."""
    groups = feature_groups(x_cal)
    scores = {name: knn_purity(value, y_cal, k=k) for name, value in groups.items()}
    selected_name = max(scores, key=scores.get)
    selected_purity = float(scores[selected_name])
    enabled = bool(selected_purity >= gate_threshold)

    # Keep a reproducible, meaningful fallback view.  Downstream MV-ACC sets
    # its RF contribution to zero when ``enabled`` is false.
    output_name = selected_name if enabled else "all24_base"
    selection = {
        "method": "CF-LCG",
        "calibration_split": "Day1 validation only",
        "k": int(k),
        "gate_threshold": float(gate_threshold),
        "candidate_purity": {name: float(score) for name, score in scores.items()},
        "best_candidate": selected_name,
        "best_purity": selected_purity,
        "rf_gate_open": enabled,
        "output_feature_group": output_name,
        "output_dimension": int(groups[output_name].shape[1]),
    }

    def extractor(x: np.ndarray) -> np.ndarray:
        return feature_groups(x)[output_name]

    return selection, extractor


def save_selection(selection: dict, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "cflcg_feature_selection.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2, ensure_ascii=False)
    return path
