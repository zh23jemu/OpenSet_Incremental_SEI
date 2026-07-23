# graph/build_graph.py
# -*- coding: utf-8 -*-

"""
Graph construction utilities for multi-view SEI / RF fingerprinting.

This module builds sample-similarity graphs from feature matrices:
    1. Deep-view graph: cosine similarity
    2. RF-view graph: Euclidean distance + RBF kernel

Input:
    features: [N, D]

Output:
    affinity graph: [N, N]
"""

import numpy as np

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def _to_numpy(x) -> np.ndarray:
    """
    Convert input to numpy array.
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    return np.asarray(x)


def _check_feature_matrix(features: np.ndarray) -> np.ndarray:
    """
    Ensure input feature matrix is [N, D].
    """
    features = _to_numpy(features).astype(np.float32)

    if features.ndim != 2:
        raise ValueError(f"Expected feature matrix [N, D], got shape {features.shape}")

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return features


def symmetrize_graph(G: np.ndarray, mode: str = "max", self_loop: bool = True) -> np.ndarray:
    """
    Symmetrize affinity graph.

    Args:
        G: [N, N] affinity matrix
        mode:
            "max": G = max(G, G.T)
            "avg": G = (G + G.T) / 2
        self_loop:
            If True, set diagonal to 1.0

    Returns:
        Symmetric graph [N, N]
    """
    G = _to_numpy(G).astype(np.float32)

    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"Expected square graph [N, N], got shape {G.shape}")

    if mode == "max":
        G = np.maximum(G, G.T)
    elif mode == "avg":
        G = 0.5 * (G + G.T)
    else:
        raise ValueError(f"Unsupported symmetrize mode: {mode}")

    G = np.nan_to_num(
        G,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    G = np.clip(G, 0.0, 1.0)

    if self_loop:
        np.fill_diagonal(G, 1.0)

    return G.astype(np.float32)


def keep_topk_affinity(
    S: np.ndarray,
    top_k: int = 20,
    self_loop: bool = True,
    symmetrize: bool = True,
) -> np.ndarray:
    """
    Keep top-k neighbors for each sample and optionally symmetrize the graph.

    Note:
        This follows your previous exp6 implementation style:
        top-k is selected directly from each row. Since self-similarity is usually
        the largest value, the diagonal/self edge is normally included.

    Args:
        S: [N, N] dense similarity matrix
        top_k: number of largest affinities to keep for each row
        self_loop: whether to set diagonal to 1
        symmetrize: whether to make graph symmetric

    Returns:
        Sparse affinity graph [N, N]
    """
    S = _to_numpy(S).astype(np.float32)

    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"Expected square affinity matrix [N, N], got shape {S.shape}")

    n = S.shape[0]

    if n == 0:
        raise ValueError("Empty affinity matrix.")

    top_k = int(top_k)

    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    top_k = min(top_k, n)

    S = np.nan_to_num(
        S,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    S = np.clip(S, 0.0, 1.0)

    S_sparse = np.zeros_like(S, dtype=np.float32)

    for i in range(n):
        idx = np.argsort(S[i])[-top_k:]
        S_sparse[i, idx] = S[i, idx]

    if symmetrize:
        S_sparse = symmetrize_graph(
            S_sparse,
            mode="max",
            self_loop=self_loop,
        )
    elif self_loop:
        np.fill_diagonal(S_sparse, 1.0)

    return S_sparse.astype(np.float32)


def build_knn_cosine_graph(
    features,
    top_k: int = 20,
    shift_to_positive: bool = True,
    self_loop: bool = True,
    symmetrize: bool = True,
) -> np.ndarray:
    """
    Build cosine-similarity kNN graph.

    This is mainly used for the deep-view graph.

    Args:
        features: [N, D]
        top_k: number of neighbors to keep
        shift_to_positive:
            If True, convert cosine similarity from [-1, 1] to [0, 1]
        self_loop: whether to set diagonal to 1
        symmetrize: whether to symmetrize the graph

    Returns:
        G_cosine: [N, N]
    """
    features = _check_feature_matrix(features)

    feat_norm = normalize(
        features,
        norm="l2",
        axis=1,
    )

    S = cosine_similarity(feat_norm)

    if shift_to_positive:
        S = (S + 1.0) / 2.0

    S = np.clip(S, 0.0, 1.0)

    G = keep_topk_affinity(
        S,
        top_k=top_k,
        self_loop=self_loop,
        symmetrize=symmetrize,
    )

    return G.astype(np.float32)


def build_knn_rbf_graph(
    features,
    top_k: int = 20,
    sigma: float = None,
    gamma: float = None,
    self_loop: bool = True,
    symmetrize: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Build RBF kNN graph from Euclidean distance.

    This is mainly used for the RF-view graph.

    RBF similarity:
        S_ij = exp(-D_ij^2 / (2 * sigma^2))

    Or if gamma is provided:
        S_ij = exp(-gamma * D_ij^2)

    Args:
        features: [N, D]
        top_k: number of neighbors to keep
        sigma: RBF bandwidth. If None, use median nonzero distance.
        gamma: Optional RBF gamma. If provided, gamma has priority over sigma.
        self_loop: whether to set diagonal to 1
        symmetrize: whether to symmetrize the graph
        eps: numerical stability

    Returns:
        G_rbf: [N, N]
    """
    features = _check_feature_matrix(features)

    D = euclidean_distances(features, features)
    D = np.nan_to_num(
        D,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if gamma is not None:
        gamma = float(gamma)
        S = np.exp(-gamma * (D ** 2))
    else:
        if sigma is None:
            nonzero = D[D > 0]

            if len(nonzero) == 0:
                sigma = 1.0
            else:
                sigma = float(np.median(nonzero))

        sigma = max(float(sigma), eps)
        S = np.exp(-(D ** 2) / (2.0 * sigma ** 2 + eps))

    S = np.clip(S, 0.0, 1.0)

    G = keep_topk_affinity(
        S,
        top_k=top_k,
        self_loop=self_loop,
        symmetrize=symmetrize,
    )

    return G.astype(np.float32)


# ---------------------------------------------------------------------
# Backward-compatible aliases.
# These names match the functions you used inside exp6.
# ---------------------------------------------------------------------

def build_cosine_graph(features, top_k: int = 20):
    """
    Alias for build_knn_cosine_graph().
    """
    return build_knn_cosine_graph(
        features=features,
        top_k=top_k,
        shift_to_positive=True,
        self_loop=True,
        symmetrize=True,
    )


def build_rbf_graph(features, top_k: int = 20, sigma: float = None):
    """
    Alias for build_knn_rbf_graph().
    """
    return build_knn_rbf_graph(
        features=features,
        top_k=top_k,
        sigma=sigma,
        gamma=None,
        self_loop=True,
        symmetrize=True,
    )


if __name__ == "__main__":
    np.random.seed(7)

    x = np.random.randn(100, 128).astype(np.float32)

    G_cos = build_knn_cosine_graph(x, top_k=20)
    G_rbf = build_knn_rbf_graph(x, top_k=20)

    print("G_cos:", G_cos.shape, G_cos.min(), G_cos.max())
    print("G_rbf:", G_rbf.shape, G_rbf.min(), G_rbf.max())