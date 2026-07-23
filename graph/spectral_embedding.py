# graph/spectral_embedding.py
# -*- coding: utf-8 -*-

"""
Spectral embedding for fused multi-view graph.

Input:
    G_fused: [N, N] precomputed affinity graph

Output:
    Z_graph: [N, m] low-dimensional graph embedding

This is used before HDBSCAN.
"""

import numpy as np

from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler, normalize


def _to_numpy(x) -> np.ndarray:
    """
    Convert input to numpy array.
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    return np.asarray(x)


def _prepare_affinity_graph(G, self_loop: bool = True) -> np.ndarray:
    """
    Clean and prepare affinity graph for SpectralEmbedding.
    """
    G = _to_numpy(G).astype(np.float32)

    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"Expected square graph [N, N], got shape {G.shape}")

    G = np.nan_to_num(
        G,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    G = np.maximum(G, G.T)
    G = np.clip(G, 0.0, 1.0)

    if self_loop:
        np.fill_diagonal(G, 1.0)

    return G.astype(np.float32)


def compute_spectral_embedding(
    G,
    n_components: int = 16,
    seed: int = 7,
    standardize: bool = True,
    l2_normalize: bool = False,
) -> np.ndarray:
    """
    Convert precomputed affinity graph into low-dimensional spectral embedding.

    Args:
        G: [N, N] affinity graph
        n_components: output graph embedding dimension
        seed: random seed
        standardize: whether to apply StandardScaler to embedding
        l2_normalize: whether to apply sample-wise L2 normalization after standardization

    Returns:
        z_graph: [N, n_components]
    """
    G = _prepare_affinity_graph(G, self_loop=True)

    n = G.shape[0]

    if n <= 2:
        raise ValueError(f"Need at least 3 samples for spectral embedding, got {n}")

    n_components = int(n_components)
    n_components = max(1, min(n_components, n - 2))

    embedder = SpectralEmbedding(
        n_components=n_components,
        affinity="precomputed",
        random_state=seed,
    )

    z_graph = embedder.fit_transform(G)

    if standardize:
        z_graph = StandardScaler().fit_transform(z_graph)

    if l2_normalize:
        z_graph = normalize(
            z_graph,
            norm="l2",
            axis=1,
        )

    z_graph = np.nan_to_num(
        z_graph,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return z_graph.astype(np.float32)


# ---------------------------------------------------------------------
# Backward-compatible alias.
# This name matches your exp6 function.
# ---------------------------------------------------------------------

def graph_to_embedding(
    G,
    n_components: int = 16,
    seed: int = 7,
):
    """
    Alias for compute_spectral_embedding().
    """
    return compute_spectral_embedding(
        G=G,
        n_components=n_components,
        seed=seed,
        standardize=True,
        l2_normalize=False,
    )


if __name__ == "__main__":
    np.random.seed(7)

    A = np.random.rand(100, 100).astype(np.float32)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 1.0)

    z = compute_spectral_embedding(
        A,
        n_components=16,
        seed=7,
    )

    print("z_graph:", z.shape)