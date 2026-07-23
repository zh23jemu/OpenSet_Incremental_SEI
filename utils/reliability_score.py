import numpy as np


def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def compute_centroid(features):
    """
    Compute centroid of a feature group.

    features: [N, D]
    """
    features = _to_numpy(features).astype(np.float32)
    if features.ndim != 2:
        raise ValueError(f"features must be [N, D], got {features.shape}")
    return np.mean(features, axis=0)


def compute_known_prototypes(features, labels, num_known_classes: int):
    """
    Compute known class prototypes.

    Args:
        features: [N, D]
        labels: [N], labels in [0, K-1]
        num_known_classes: K

    Returns:
        prototypes: [K, D]
    """
    features = _to_numpy(features).astype(np.float32)
    labels = _to_numpy(labels).astype(np.int64)

    prototypes = []

    for k in range(num_known_classes):
        idx = labels == k
        if idx.sum() == 0:
            raise ValueError(f"No samples found for known class {k}.")
        prototypes.append(np.mean(features[idx], axis=0))

    return np.stack(prototypes, axis=0).astype(np.float32)


def distance_to_known(centroid, known_prototypes):
    """
    Compute minimum distance from one cluster centroid to known prototypes.
    """
    centroid = _to_numpy(centroid).astype(np.float32)
    known_prototypes = _to_numpy(known_prototypes).astype(np.float32)

    distances = np.linalg.norm(known_prototypes - centroid[None, :], axis=1)
    return float(np.min(distances))


def compute_compactness(cluster_features, eps: float = 1e-8):
    """
    Compactness score.

    Higher means more compact.
    Range approximately: (0, 1]

    R_compact = exp(- mean distance to centroid)
    """
    cluster_features = _to_numpy(cluster_features).astype(np.float32)

    if cluster_features.shape[0] <= 1:
        return 0.0

    centroid = np.mean(cluster_features, axis=0)
    dist = np.linalg.norm(cluster_features - centroid[None, :], axis=1)
    mean_dist = np.mean(dist)

    score = np.exp(-mean_dist)
    return float(score)


def compute_separation(cluster_id, cluster_labels, features):
    """
    Separation score.

    Higher means farther from other clusters.
    If there is only one valid cluster, return 1.0.
    """
    features = _to_numpy(features).astype(np.float32)
    cluster_labels = _to_numpy(cluster_labels)

    valid_clusters = [c for c in np.unique(cluster_labels) if c != -1]

    if cluster_id not in valid_clusters:
        raise ValueError(f"cluster_id {cluster_id} not found in valid clusters.")

    if len(valid_clusters) <= 1:
        return 1.0

    current_idx = cluster_labels == cluster_id
    current_centroid = np.mean(features[current_idx], axis=0)

    other_distances = []

    for c in valid_clusters:
        if c == cluster_id:
            continue
        idx = cluster_labels == c
        centroid = np.mean(features[idx], axis=0)
        dist = np.linalg.norm(current_centroid - centroid)
        other_distances.append(dist)

    min_dist = np.min(other_distances)

    score = 1.0 - np.exp(-min_dist)
    return float(score)


def compute_rf_consistency(cluster_rf_features):
    """
    RF consistency score.

    This is similar to compactness, but computed in classical RF feature space.

    Higher means RF features inside the cluster are more consistent.
    """
    cluster_rf_features = _to_numpy(cluster_rf_features).astype(np.float32)

    if cluster_rf_features.shape[0] <= 1:
        return 0.0

    centroid = np.mean(cluster_rf_features, axis=0)
    dist = np.linalg.norm(cluster_rf_features - centroid[None, :], axis=1)
    mean_dist = np.mean(dist)

    score = np.exp(-mean_dist)
    return float(score)


def compute_slot_consistency(cluster_slot_indices):
    """
    VUP slot consistency.

    Args:
        cluster_slot_indices: [N_cluster]
            each element is the strongest VUP slot index of one sample

    Returns:
        R_slot: max slot ratio in this cluster

    Example:
        VUP1: 460 samples
        VUP2: 20 samples
        VUP3: 20 samples
        R_slot = 460 / 500 = 0.92
    """
    cluster_slot_indices = _to_numpy(cluster_slot_indices).astype(np.int64)

    if len(cluster_slot_indices) == 0:
        return 0.0

    _, counts = np.unique(cluster_slot_indices, return_counts=True)
    return float(np.max(counts) / len(cluster_slot_indices))


def geometric_mean(values, eps: float = 1e-8):
    """
    Geometric mean for reliability scoring.

    values should be in [0, 1].
    """
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return 0.0

    values = np.clip(values, eps, 1.0)
    return float(np.exp(np.mean(np.log(values))))


def evaluate_cluster_reliability(
    cluster_id,
    cluster_labels,
    features,
    known_prototypes=None,
    unknown_scores=None,
    rf_features=None,
    slot_indices=None,
    min_cluster_size: int = 30,
    tau_known: float = None,
    tau_conf: float = None,
    tau_reliable: float = 0.5,
):
    """
    Evaluate reliability score for one HDBSCAN cluster.

    Args:
        cluster_id: HDBSCAN cluster id, excluding -1
        cluster_labels: [N]
        features: [N, D], features used for clustering
        known_prototypes: [K, D], optional
        unknown_scores: [N], optional VUP unknown scores
        rf_features: [N, D_rf], optional classical RF features
        slot_indices: [N], optional strongest VUP slot index
        min_cluster_size: minimum size for hard filtering
        tau_known: distance-to-known threshold
        tau_conf: unknown confidence threshold
        tau_reliable: final reliability threshold

    Returns:
        result: dict
    """
    cluster_labels = _to_numpy(cluster_labels)
    features = _to_numpy(features).astype(np.float32)

    idx = cluster_labels == cluster_id
    cluster_features = features[idx]
    cluster_size = int(idx.sum())

    result = {
        "cluster_id": int(cluster_id),
        "cluster_size": cluster_size,
        "hard_pass": True,
        "hard_fail_reasons": [],
        "D_known": None,
        "R_conf": None,
        "R_compact": None,
        "R_sep": None,
        "R_rf": None,
        "R_slot": None,
        "R_soft": None,
        "R_final": None,
        "decision": None,
    }

    # ------------------------------------------------------------
    # Hard Filtering 1: cluster size
    # ------------------------------------------------------------
    if cluster_size < min_cluster_size:
        result["hard_pass"] = False
        result["hard_fail_reasons"].append("cluster_size_too_small")

    # ------------------------------------------------------------
    # Hard Filtering 2: distance to known prototypes
    # ------------------------------------------------------------
    if known_prototypes is not None and tau_known is not None:
        centroid = compute_centroid(cluster_features)
        d_known = distance_to_known(centroid, known_prototypes)
        result["D_known"] = d_known

        if d_known <= tau_known:
            result["hard_pass"] = False
            result["hard_fail_reasons"].append("too_close_to_known")

    # ------------------------------------------------------------
    # Hard Filtering 3: unknown confidence
    # ------------------------------------------------------------
    if unknown_scores is not None and tau_conf is not None:
        unknown_scores = _to_numpy(unknown_scores).astype(np.float32)
        r_conf = float(np.mean(unknown_scores[idx]))
        result["R_conf"] = r_conf

        if r_conf <= tau_conf:
            result["hard_pass"] = False
            result["hard_fail_reasons"].append("unknown_confidence_too_low")

    # ------------------------------------------------------------
    # Soft Scoring
    # ------------------------------------------------------------
    r_compact = compute_compactness(cluster_features)
    r_sep = compute_separation(cluster_id, cluster_labels, features)

    result["R_compact"] = r_compact
    result["R_sep"] = r_sep

    soft_scores = [r_compact, r_sep]

    # Optional RF consistency
    if rf_features is not None:
        rf_features = _to_numpy(rf_features).astype(np.float32)
        cluster_rf_features = rf_features[idx]
        r_rf = compute_rf_consistency(cluster_rf_features)
        result["R_rf"] = r_rf
        soft_scores.append(r_rf)

    # Optional VUP slot consistency
    if slot_indices is not None:
        slot_indices = _to_numpy(slot_indices).astype(np.int64)
        cluster_slot = slot_indices[idx]
        r_slot = compute_slot_consistency(cluster_slot)
        result["R_slot"] = r_slot
        soft_scores.append(r_slot)

    r_soft = geometric_mean(soft_scores)
    result["R_soft"] = r_soft

    # ------------------------------------------------------------
    # Final Reliability Score
    # ------------------------------------------------------------
    if result["hard_pass"]:
        r_final = r_soft
    else:
        r_final = 0.0

    result["R_final"] = r_final

    if r_final >= tau_reliable:
        result["decision"] = "reliable"
    elif result["hard_pass"]:
        result["decision"] = "pending"
    else:
        result["decision"] = "reject"

    return result


def compute_reliability_scores(
    cluster_labels,
    features,
    known_prototypes=None,
    unknown_scores=None,
    rf_features=None,
    slot_indices=None,
    min_cluster_size: int = 30,
    tau_known: float = None,
    tau_conf: float = None,
    tau_reliable: float = 0.5,
):
    """
    Compute reliability scores for all non-noise clusters.

    Returns:
        results: list of dict
    """
    cluster_labels = _to_numpy(cluster_labels)

    valid_clusters = [c for c in np.unique(cluster_labels) if c != -1]

    results = []

    for cluster_id in valid_clusters:
        result = evaluate_cluster_reliability(
            cluster_id=cluster_id,
            cluster_labels=cluster_labels,
            features=features,
            known_prototypes=known_prototypes,
            unknown_scores=unknown_scores,
            rf_features=rf_features,
            slot_indices=slot_indices,
            min_cluster_size=min_cluster_size,
            tau_known=tau_known,
            tau_conf=tau_conf,
            tau_reliable=tau_reliable,
        )
        results.append(result)

    return results


def print_reliability_results(results):
    """
    Pretty print reliability score results.
    """
    print("\n" + "=" * 80)
    print("Reliability Score Results")
    print("=" * 80)

    for r in results:
        print(
            f"Cluster {r['cluster_id']} | "
            f"size={r['cluster_size']} | "
            f"hard_pass={r['hard_pass']} | "
            f"R_final={r['R_final']:.4f} | "
            f"decision={r['decision']}"
        )

        print(
            f"  R_compact={r['R_compact']:.4f}, "
            f"R_sep={r['R_sep']:.4f}, "
            f"R_rf={r['R_rf']}, "
            f"R_slot={r['R_slot']}"
        )

        if r["D_known"] is not None:
            print(f"  D_known={r['D_known']:.4f}")

        if r["R_conf"] is not None:
            print(f"  R_conf={r['R_conf']:.4f}")

        if len(r["hard_fail_reasons"]) > 0:
            print(f"  fail_reasons={r['hard_fail_reasons']}")