"""Validation-only benchmark for generic versus ADS-B-specific RF features."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.adsb_90known_strict_loader import _as_int_labels, _load_npy, _take_and_convert
from features.adsb_features import extract_adsb_feature_groups
from utils.classic_feature_gating import feature_groups, knn_purity


def validation_only(data_root):
    root = Path(data_root).resolve()
    x_mmap = _load_npy(root, "X_train_90Class.npy")
    y = _as_int_labels(_load_npy(root, "Y_train_90Class.npy"))
    validation_indices = []
    for class_id in range(90):
        indices = np.flatnonzero(y == class_id)
        train_end = int(np.floor(0.60 * len(indices)))
        val_end = int(np.floor(0.70 * len(indices)))
        validation_indices.append(indices[train_end:val_end])
    validation_indices = np.concatenate(validation_indices)
    return _take_and_convert(x_mmap, validation_indices), y[validation_indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=r"C:\Users\123\Downloads\Dataset\Dataset")
    parser.add_argument("--save_dir", default="./results/adsb_classic_feature_dev")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    x_val, y_val = validation_only(args.data_root)
    candidates = {}
    candidates.update({f"generic_{k}": v for k, v in feature_groups(x_val).items()})
    candidates.update(extract_adsb_feature_groups(x_val))

    rows = []
    for name, values in candidates.items():
        row = {
            "feature_group": name,
            "dimension": int(values.shape[1]),
            "purity_k5": knn_purity(values, y_val, k=5),
            "purity_k10": knn_purity(values, y_val, k=10),
            "purity_k20": knn_purity(values, y_val, k=20),
        }
        row["mean_purity"] = np.mean([row["purity_k5"], row["purity_k10"], row["purity_k20"]])
        rows.append(row)
        print(row)
    rows.sort(key=lambda row: row["mean_purity"], reverse=True)
    report = {
        "protocol": "base file stored-order validation block only (60% train / 10% validation / 30% untouched)",
        "validation_samples": int(len(y_val)),
        "classes": 90,
        "test_files_opened": False,
        "payload_bits_or_labels_used_by_features": False,
        "ranking": rows,
    }
    with open(os.path.join(args.save_dir, "adsb_feature_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    best = rows[0]
    print(f"[Best] {best['feature_group']} dim={best['dimension']} mean_purity={best['mean_purity']:.4f}")


if __name__ == "__main__":
    main()
