import numpy as np

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score


def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def standardize_features(features, eps: float = 1e-8):
    """
    Standardize features before HDBSCAN.

    This is important because HDBSCAN is distance-based.
    """
    features = _to_numpy(features).astype(np.float32)

    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)

    out = (features - mean) / (std + eps)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    return out.astype(np.float32)


def l2_normalize_features(features, eps: float = 1e-8):
    """
    L2 normalize feature vectors.
    """
    features = _to_numpy(features).astype(np.float32)
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / (norm + eps)


def run_hdbscan(
    features,
    min_cluster_size: int = 30,
    min_samples: int = 10,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    standardize: bool = True,
):
    """
    Run HDBSCAN on feature vectors.

    Args:
        features: [N, D]
        min_cluster_size: minimum cluster size
        min_samples: conservative density parameter
        metric: distance metric
        cluster_selection_method: "eom" or "leaf"
        standardize: whether to standardize features before clustering

    Returns:
        cluster_labels: [N], -1 means noise
        clusterer: fitted HDBSCAN object
    """
    try:
        import hdbscan
    except ImportError as e:
        raise ImportError(
            "Package `hdbscan` is not installed. "
            "Install it with: pip install hdbscan"
        ) from e

    features = _to_numpy(features).astype(np.float32)

    if features.ndim != 2:
        raise ValueError(f"features must be [N, D], but got {features.shape}")

    if standardize:
        features_for_cluster = standardize_features(features)
    else:
        features_for_cluster = features

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )

    cluster_labels = clusterer.fit_predict(features_for_cluster)

    return cluster_labels, clusterer


def cluster_purity(y_true, cluster_labels, ignore_noise: bool = True):
    """
    Calculate cluster purity.

    Args:
        y_true: true labels, [N]
        cluster_labels: predicted cluster labels, [N], -1 means noise
        ignore_noise: whether to ignore noise points

    Returns:
        purity: float
    """
    y_true = _to_numpy(y_true)
    cluster_labels = _to_numpy(cluster_labels)

    if y_true.shape[0] != cluster_labels.shape[0]:
        raise ValueError("y_true and cluster_labels must have same length.")

    if ignore_noise:
        mask = cluster_labels != -1
        y_true = y_true[mask]
        cluster_labels = cluster_labels[mask]

    if len(y_true) == 0:
        return 0.0

    total_correct = 0
    total_count = 0

    for c in np.unique(cluster_labels):
        idx = cluster_labels == c
        if idx.sum() == 0:
            continue

        labels, counts = np.unique(y_true[idx], return_counts=True)
        total_correct += counts.max()
        total_count += idx.sum()

    if total_count == 0:
        return 0.0

    return float(total_correct / total_count)


def calculate_nmi_ari_cluster_purity(
    y_true,
    cluster_labels,
    ignore_noise: bool = True,
):
    """
    Calculate NMI, ARI and cluster purity.

    Args:
        y_true: true labels
        cluster_labels: HDBSCAN cluster labels
        ignore_noise: ignore label -1 when computing metrics

    Returns:
        nmi, ari, purity
    """
    y_true = _to_numpy(y_true)
    cluster_labels = _to_numpy(cluster_labels)

    if ignore_noise:
        mask = cluster_labels != -1
        y_eval = y_true[mask]
        c_eval = cluster_labels[mask]
    else:
        y_eval = y_true
        c_eval = cluster_labels

    if len(y_eval) == 0:
        return 0.0, 0.0, 0.0

    nmi = normalized_mutual_info_score(y_eval, c_eval)
    ari = adjusted_rand_score(y_eval, c_eval)
    purity = cluster_purity(y_eval, c_eval, ignore_noise=False)

    return float(nmi), float(ari), float(purity)


def summarize_clustering(y_true, cluster_labels, ignore_noise: bool = True):
    """
    Return a dictionary of clustering results.
    """
    y_true = _to_numpy(y_true)
    cluster_labels = _to_numpy(cluster_labels)

    nmi, ari, purity = calculate_nmi_ari_cluster_purity(
        y_true,
        cluster_labels,
        ignore_noise=ignore_noise,
    )

    total_samples = len(cluster_labels)
    noise_count = int(np.sum(cluster_labels == -1))
    clustered_samples = int(total_samples - noise_count)

    valid_clusters = sorted([c for c in np.unique(cluster_labels) if c != -1])
    num_clusters = len(valid_clusters)

    return {
        "NMI_clustered_only": nmi,
        "ARI_clustered_only": ari,
        "Cluster_Purity_clustered_only": purity,
        "Num_Clusters": num_clusters,
        "Noise_Ratio": float(noise_count / total_samples) if total_samples > 0 else 0.0,
        "Clustered_Samples": clustered_samples,
        "Total_Samples": int(total_samples),
    }


def print_clustering_summary(summary: dict, title: str = "Clustering Summary"):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for k, v in summary.items():
        print(f"{k}: {v}")