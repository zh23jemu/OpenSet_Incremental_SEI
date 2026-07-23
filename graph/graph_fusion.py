# graph/graph_fusion.py
# -*- coding: utf-8 -*-

"""
Multi-view graph fusion utilities.

Main formula:
    G_fused = alpha * G_deep + (1 - alpha) * G_rf

This module also provides build_multiview_graph(), which directly builds:
    Deep-view graph
    RF-view graph
    Fused graph
"""

import numpy as np

try:
    from .build_graph import (
        build_knn_cosine_graph,
        build_knn_rbf_graph,
        symmetrize_graph,
    )
except ImportError:
    from build_graph import (
        build_knn_cosine_graph,
        build_knn_rbf_graph,
        symmetrize_graph,
    )


def _to_numpy(x) -> np.ndarray:
    """
    Convert input to numpy array.
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    return np.asarray(x)


def _check_square_graph(G: np.ndarray, name: str = "G") -> np.ndarray:
    """
    Check graph shape and clean invalid values.
    """
    G = _to_numpy(G).astype(np.float32)

    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"{name} must be a square matrix [N, N], got shape {G.shape}")

    G = np.nan_to_num(
        G,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    G = np.clip(G, 0.0, 1.0)

    return G.astype(np.float32)


def normalize_graph(
    G,
    mode: str = "none",
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize graph before fusion.

    Args:
        G: [N, N] affinity graph
        mode:
            "none": no normalization
            "max": divide by max value
            "row": row-normalization
            "symmetric": D^{-1/2} G D^{-1/2}
        eps: numerical stability

    Returns:
        normalized graph [N, N]
    """
    G = _check_square_graph(G)

    if mode is None:
        mode = "none"

    mode = mode.lower()

    if mode == "none":
        return G.astype(np.float32)

    if mode == "max":
        max_val = float(np.max(G))
        if max_val > eps:
            G = G / max_val
        return np.clip(G, 0.0, 1.0).astype(np.float32)

    if mode == "row":
        row_sum = np.sum(G, axis=1, keepdims=True)
        G = G / (row_sum + eps)
        return np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if mode == "symmetric":
        degree = np.sum(G, axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(degree + eps)
        G_norm = d_inv_sqrt[:, None] * G * d_inv_sqrt[None, :]
        G_norm = np.nan_to_num(
            G_norm,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return G_norm.astype(np.float32)

    raise ValueError(f"Unsupported graph normalization mode: {mode}")


def fuse_graphs(
    G_deep,
    G_rf,
    alpha: float = 0.7,
    normalize_mode: str = "none",
    self_loop: bool = True,
) -> np.ndarray:
    """
    Fuse deep-view graph and RF-view graph.

    Args:
        G_deep: [N, N] deep-view affinity graph
        G_rf: [N, N] RF-view affinity graph
        alpha:
            fusion weight.
            alpha = 1.0 means deep-only graph.
            alpha = 0.0 means RF-only graph.
        normalize_mode:
            graph normalization before fusion.
            To reproduce your previous exp6 behavior, keep "none".
        self_loop: whether to set diagonal to 1

    Returns:
        G_fused: [N, N]
    """
    G_deep = _check_square_graph(G_deep, name="G_deep")
    G_rf = _check_square_graph(G_rf, name="G_rf")

    if G_deep.shape != G_rf.shape:
        raise ValueError(
            f"G_deep and G_rf must have the same shape, "
            f"got {G_deep.shape} and {G_rf.shape}"
        )

    alpha = float(alpha)

    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    G_deep = normalize_graph(G_deep, mode=normalize_mode)
    G_rf = normalize_graph(G_rf, mode=normalize_mode)

    G_fused = alpha * G_deep + (1.0 - alpha) * G_rf
    G_fused = np.clip(G_fused, 0.0, 1.0)

    G_fused = symmetrize_graph(
        G_fused,
        mode="max",
        self_loop=self_loop,
    )

    return G_fused.astype(np.float32)


def build_multiview_graph(
    z_deep_norm,
    z_rf_norm,
    alpha: float = 0.7,
    top_k: int = 20,
    deep_graph_type: str = "cosine",
    rf_graph_type: str = "rbf",
    normalize_mode: str = "none",
):
    """
    Build deep-view graph, RF-view graph, and fused graph.

    Args:
        z_deep_norm: [N, D_deep], usually standardized deep embedding
        z_rf_norm: [N, D_rf], standardized RF features
        alpha: deep graph fusion weight
        top_k: top-k neighbors for each view graph
        deep_graph_type: "cosine" or "rbf"
        rf_graph_type: "cosine" or "rbf"
        normalize_mode: graph normalization before fusion

    Returns:
        G_fused: [N, N]
        G_deep: [N, N]
        G_rf: [N, N]
    """
    deep_graph_type = deep_graph_type.lower()
    rf_graph_type = rf_graph_type.lower()

    if deep_graph_type == "cosine":
        G_deep = build_knn_cosine_graph(
            z_deep_norm,
            top_k=top_k,
            shift_to_positive=True,
            self_loop=True,
            symmetrize=True,
        )
    elif deep_graph_type == "rbf":
        G_deep = build_knn_rbf_graph(
            z_deep_norm,
            top_k=top_k,
            sigma=None,
            gamma=None,
            self_loop=True,
            symmetrize=True,
        )
    else:
        raise ValueError(f"Unsupported deep_graph_type: {deep_graph_type}")

    if rf_graph_type == "cosine":
        G_rf = build_knn_cosine_graph(
            z_rf_norm,
            top_k=top_k,
            shift_to_positive=True,
            self_loop=True,
            symmetrize=True,
        )
    elif rf_graph_type == "rbf":
        G_rf = build_knn_rbf_graph(
            z_rf_norm,
            top_k=top_k,
            sigma=None,
            gamma=None,
            self_loop=True,
            symmetrize=True,
        )
    else:
        raise ValueError(f"Unsupported rf_graph_type: {rf_graph_type}")

    G_fused = fuse_graphs(
        G_deep=G_deep,
        G_rf=G_rf,
        alpha=alpha,
        normalize_mode=normalize_mode,
        self_loop=True,
    )

    return G_fused.astype(np.float32), G_deep.astype(np.float32), G_rf.astype(np.float32)


if __name__ == "__main__":
    np.random.seed(7)

    z_deep = np.random.randn(100, 128).astype(np.float32)
    z_rf = np.random.randn(100, 24).astype(np.float32)

    G, G_deep, G_rf = build_multiview_graph(
        z_deep_norm=z_deep,
        z_rf_norm=z_rf,
        alpha=0.7,
        top_k=20,
    )

    print("G_deep:", G_deep.shape)
    print("G_rf:", G_rf.shape)
    print("G_fused:", G.shape)