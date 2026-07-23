"""Small paper-facing visualizations for the complete seen-class state."""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt


def _project(features, method: str, seed: int):
    x = np.nan_to_num(np.asarray(features, dtype=np.float32))
    if method == "umap":
        try:
            import umap
            return umap.UMAP(n_components=2, n_neighbors=min(20, max(5, len(x) - 1)), min_dist=0.12,
                             metric="euclidean", random_state=seed).fit_transform(x)
        except ImportError as exc:
            raise RuntimeError("UMAP visualization requires umap-learn. Install it with: pip install umap-learn") from exc
    from sklearn.manifold import TSNE
    perplexity = min(30, max(5, (len(x) - 1) // 3))
    return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed).fit_transform(x)


def _plot(points, labels, title, path):
    labels = np.asarray(labels)
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("tab20", max(20, int(np.unique(labels).size)))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(points[idx, 0], points[idx, 1], s=18, alpha=0.82,
                    color=cmap(int(label) % cmap.N), label=f"Tx {int(label)}")
    plt.title(title, fontsize=15)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    if len(np.unique(labels)) <= 20:
        plt.legend(ncol=2, fontsize=7, markerscale=1.2, frameon=True)
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def save_seen_class_visualizations(features, true_labels, operational_labels, save_dir, round_name, seed):
    """Save UMAP/t-SNE separately for all classes seen through one round.

    ``true_labels`` are display-only.  ``operational_labels`` contain known
    class IDs plus the discovered pseudo-label IDs, avoiding label leakage.
    """
    for method in ("umap", "tsne"):
        out_dir = os.path.join(save_dir, "paper_visualizations", round_name.lower(), method)
        os.makedirs(out_dir, exist_ok=True)
        points = _project(features, method, seed)
        name = method.upper() if method == "umap" else "t-SNE"
        _plot(points, true_labels, f"MV-ACC {round_name}: seen Tx true labels ({name})",
              os.path.join(out_dir, "seen_true_labels.png"))
        _plot(points, operational_labels, f"MV-ACC {round_name}: known classes + discovered pseudo-labels ({name})",
              os.path.join(out_dir, "seen_operational_labels.png"))
        np.savetxt(os.path.join(out_dir, "embedding_2d.csv"),
                   np.column_stack([points, true_labels, operational_labels]), delimiter=",",
                   header="dim1,dim2,true_label,operational_label", comments="")


def save_paper_method_visualizations(method, round_name, known_features, known_labels,
                                     new_features, new_cluster_labels, save_dir, seed,
                                     metrics=None):
    """Save colour UMAP/t-SNE views of the full incremental state.

    Circles are all previously registered classes (including Day-1 and classes
    discovered in earlier rounds).  Current-round discovery samples retain the
    cluster colour and their cluster centroids are shown by large star markers.
    No ground-truth unknown labels are used in the plot construction.
    """
    known_features = np.asarray(known_features)
    new_features = np.asarray(new_features)
    known_labels = np.asarray(known_labels)
    new_cluster_labels = np.asarray(new_cluster_labels)
    features = np.concatenate([known_features, new_features], axis=0)
    all_labels = np.concatenate([known_labels, new_cluster_labels])
    known_n = len(known_labels)
    out_dir = os.path.join(save_dir, "paper_visualizations", "all_methods",
                           method.lower().replace(" ", "_").replace("(", "").replace(")", ""),
                           round_name.lower())
    os.makedirs(out_dir, exist_ok=True)

    unique_labels = list(np.unique(all_labels))
    colour_map = plt.get_cmap("tab20", max(20, len(unique_labels)))
    colour_for = {label: colour_map(i % colour_map.N) for i, label in enumerate(unique_labels)}
    metric_text = ""
    if metrics:
        k = metrics.get("Final Cluster Count", metrics.get("Cluster Count", "-"))
        nmi = metrics.get("NMI", None)
        ari = metrics.get("ARI", None)
        metric_text = f"  |  K={k}"
        if nmi is not None:
            metric_text += f", NMI={float(nmi):.3f}"
        if ari is not None:
            metric_text += f", ARI={float(ari):.3f}"

    for projection in ("umap", "tsne"):
        points = _project(features, projection, seed)
        fig, ax = plt.subplots(figsize=(11.2, 8.3))
        # Every known/registered class remains coloured.  We intentionally do
        # not create a 10-30 entry known-class legend, because it obscures the data.
        for label in np.unique(known_labels):
            idx = np.where(known_labels == label)[0]
            ax.scatter(points[idx, 0], points[idx, 1], s=13, alpha=0.58,
                       color=colour_for[label], edgecolors="none", rasterized=True)
        # Current-round points use stronger colours; the large star is the
        # operational marker for a newly discovered/enrolled device cluster.
        for label in np.unique(new_cluster_labels):
            idx = np.where(new_cluster_labels == label)[0]
            p = points[known_n + idx]
            col = colour_for[label]
            ax.scatter(p[:, 0], p[:, 1], s=25, alpha=0.90, color=col,
                       edgecolors="white", linewidths=0.25, rasterized=True)
            center = p.mean(axis=0)
            ax.scatter(center[0], center[1], s=260, marker="*", color=col,
                       edgecolors="#111111", linewidths=0.8, zorder=8)

        name = "UMAP" if projection == "umap" else "t-SNE"
        ax.set_title(f"{method} · {round_name}: registered classes + current discovery ({name}){metric_text}",
                     fontsize=15, pad=12, fontweight="bold")
        ax.set_xlabel(f"{name}-1", fontsize=11)
        ax.set_ylabel(f"{name}-2", fontsize=11)
        ax.grid(alpha=0.12, linewidth=0.6)
        ax.text(0.01, 0.01, "● coloured circles: known / previously enrolled classes\n★ coloured stars: current-round discovered-cluster centroids",
                transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.86, edgecolor="#BBBBBB"))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"known_and_incremental_{projection}.png"), dpi=320, bbox_inches="tight")
        plt.close(fig)
        np.savetxt(os.path.join(out_dir, f"embedding_{projection}.csv"),
                   np.column_stack([points, all_labels, np.r_[np.zeros(known_n), np.ones(len(new_cluster_labels))]]),
                   delimiter=",", header="dim1,dim2,operational_label,is_current_round", comments="")
