import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auroc(y_true, scores):
    """
    y_true: 0 = known, 1 = unknown
    scores: larger score means more likely to be unknown
    """
    return float(roc_auc_score(y_true, scores))


def compute_fpr95(y_true, scores):
    """
    FPR when TPR reaches 95%.
    Lower is better.
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.where(tpr >= 0.95)[0]

    if len(idx) == 0:
        return 1.0

    return float(fpr[idx[0]])


def compute_open_set_stats(y_true, scores, threshold):
    """
    y_true: 0 = known, 1 = unknown
    scores: unknown scores
    threshold: samples with score >= threshold enter unknown buffer
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    pred_unknown = scores >= threshold

    total_unknown = np.sum(y_true == 1)
    true_unknown_in_buffer = np.sum((y_true == 1) & pred_unknown)
    known_in_buffer = np.sum((y_true == 0) & pred_unknown)
    unknown_to_known = np.sum((y_true == 1) & (~pred_unknown))

    buffer_total = true_unknown_in_buffer + known_in_buffer

    return {
        "unknown_to_known_error": float(unknown_to_known / max(total_unknown, 1)),
        "buffer_purity": float(true_unknown_in_buffer / max(buffer_total, 1)),
        "buffer_recall": float(true_unknown_in_buffer / max(total_unknown, 1)),
        "buffer_total": int(buffer_total),
        "true_unknown_in_buffer": int(true_unknown_in_buffer),
        "known_in_buffer": int(known_in_buffer),
    }