"""ADS-B 90 -> 100 -> 110 -> 120 staged open-set incremental loader.

The downloaded corpus contains two disjoint label spaces:

* ``X_train_90Class.npy`` / ``Y_train_90Class.npy``: 90 labeled base
  aircraft identities used to train the closed-set representation.
* ``X_train_30Class.npy`` + ``X_test_30Class.npy``: a separate pool of 30
  novel aircraft identities with provided train/test partitions.

The numeric labels in both pools start at zero, so the novel labels MUST be
offset.  This loader keeps base-aircraft labels at 0..89 and maps novel
aircraft to 90..119.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


DEFAULT_ADSB_ROOT = r"C:\Users\123\Downloads\Dataset\Dataset"


def _load_npy(root: Path, name: str, mmap_mode="r"):
    path = root / name
    if not path.exists():
        raise FileNotFoundError(f"Missing ADS-B file: {path}")
    return np.load(path, mmap_mode=mmap_mode)


def _as_int_labels(values) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 2 and values.shape[1] > 1:
        values = np.argmax(values, axis=1)
    return values.reshape(-1).astype(np.int64)


def _to_n2l_rms_normalized(values, chunk_size: int = 256) -> np.ndarray:
    """Convert [N,L,2] or [N,2,L] float arrays to [N,2,L] float32.

    A single complex RMS scale is used per instance.  Unlike independent I/Q
    z-scoring, this preserves DC offset and I/Q imbalance cues used by the RF
    handcrafted view.
    """
    values = np.asarray(values)
    if values.ndim != 3:
        raise ValueError(f"Expected three-dimensional IQ array, got {values.shape}")
    if values.shape[-1] == 2:
        n, length, _ = values.shape
        source_is_nl2 = True
    elif values.shape[1] == 2:
        n, _, length = values.shape
        source_is_nl2 = False
    else:
        raise ValueError(f"Unsupported IQ shape: {values.shape}")

    output = np.empty((n, 2, length), dtype=np.float32)
    for begin in range(0, n, chunk_size):
        end = min(begin + chunk_size, n)
        block = np.asarray(values[begin:end], dtype=np.float32)
        if source_is_nl2:
            block = np.transpose(block, (0, 2, 1))
        energy = np.mean(block[:, 0] ** 2 + block[:, 1] ** 2, axis=1, keepdims=True)
        rms = np.sqrt(np.maximum(energy, 1e-12))[:, :, None]
        output[begin:end] = block / rms
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def _take_and_convert(x_mmap, indices: np.ndarray) -> np.ndarray:
    # Advanced indexing happens once per split and loads only selected samples.
    selected = np.asarray(x_mmap[indices])
    return _to_n2l_rms_normalized(selected)


def _class_stratified_known_split(
    x_mmap,
    y: np.ndarray,
    selected_original_classes: Sequence[int],
    train_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between zero and one")
    train_indices: List[np.ndarray] = []
    eval_indices: List[np.ndarray] = []
    train_labels: List[np.ndarray] = []
    eval_labels: List[np.ndarray] = []
    split_rows = []

    for new_label, original_label in enumerate(selected_original_classes):
        indices = np.flatnonzero(y == int(original_label))
        if len(indices) < 2:
            raise ValueError(f"Known class {original_label} has fewer than two samples")
        rng = np.random.default_rng(seed + 1009 * int(original_label))
        indices = indices[rng.permutation(len(indices))]
        cut = min(max(int(round(len(indices) * train_ratio)), 1), len(indices) - 1)
        tr, ev = indices[:cut], indices[cut:]
        train_indices.append(tr)
        eval_indices.append(ev)
        train_labels.append(np.full(len(tr), new_label, dtype=np.int64))
        eval_labels.append(np.full(len(ev), new_label, dtype=np.int64))
        split_rows.append(
            {
                "original_label": int(original_label),
                "global_label": int(new_label),
                "total": int(len(indices)),
                "train": int(len(tr)),
                "eval": int(len(ev)),
            }
        )

    tr_idx = np.concatenate(train_indices)
    ev_idx = np.concatenate(eval_indices)
    if np.intersect1d(tr_idx, ev_idx).size:
        raise RuntimeError("Known train/evaluation index overlap detected")
    return (
        _take_and_convert(x_mmap, tr_idx),
        np.concatenate(train_labels),
        _take_and_convert(x_mmap, ev_idx),
        np.concatenate(eval_labels),
        {"per_class": split_rows, "index_overlap": 0},
    )


def _select_unknown_group(
    x_mmap,
    y_local: np.ndarray,
    local_classes: Iterable[int],
    global_offset: int,
) -> Tuple[np.ndarray, np.ndarray]:
    local_classes = np.asarray(list(local_classes), dtype=np.int64)
    mask = np.isin(y_local, local_classes)
    indices = np.flatnonzero(mask)
    x = _take_and_convert(x_mmap, indices)
    y = y_local[indices] + int(global_offset)
    return x, y.astype(np.int64)


def load_adsb_90known_3round(
    data_root: str = DEFAULT_ADSB_ROOT,
    initial_known_classes: int = 90,
    round_size: int = 10,
    num_rounds: int = 3,
    train_ratio: float = 0.70,
    known_selection_seed: int = 2026,
    split_seed: int = 2026,
) -> dict:
    """Load the dataset-native 90-known + 3x10-unknown ADS-B protocol."""
    if (initial_known_classes, round_size, num_rounds) != (90, 10, 3):
        raise ValueError("This loader is specialized for ADS-B 90 -> 100 -> 110 -> 120")

    root = Path(data_root).expanduser().resolve()
    x_known = _load_npy(root, "X_train_90Class.npy")
    y_known_original = _as_int_labels(_load_npy(root, "Y_train_90Class.npy"))
    x_unknown_train = _load_npy(root, "X_train_30Class.npy")
    y_unknown_train = _as_int_labels(_load_npy(root, "Y_train_30Class.npy"))
    x_unknown_test = _load_npy(root, "X_test_30Class.npy")
    y_unknown_test = _as_int_labels(_load_npy(root, "Y_test_30Class.npy"))

    if len(x_known) != len(y_known_original):
        raise ValueError("Known X/Y length mismatch")
    if len(x_unknown_train) != len(y_unknown_train):
        raise ValueError("Unknown-train X/Y length mismatch")
    if len(x_unknown_test) != len(y_unknown_test):
        raise ValueError("Unknown-test X/Y length mismatch")

    available_known = np.unique(y_known_original)
    available_unknown_train = np.unique(y_unknown_train)
    available_unknown_test = np.unique(y_unknown_test)
    if not np.array_equal(available_known, np.arange(90)):
        raise ValueError(f"Unexpected base-pool labels: {available_known.tolist()}")
    if not np.array_equal(available_unknown_train, np.arange(30)):
        raise ValueError(f"Unexpected unknown-train labels: {available_unknown_train.tolist()}")
    if not np.array_equal(available_unknown_test, np.arange(30)):
        raise ValueError(f"Unexpected unknown-test labels: {available_unknown_test.tolist()}")

    # Dataset-native protocol: all 90 labeled base identities are used.  Keep
    # the argument in the metadata/API for compatibility and reproducibility.
    selected_known_original = available_known.astype(int).tolist()
    x_train, y_train, x_known_eval, y_known_eval, known_audit = _class_stratified_known_split(
        x_known,
        y_known_original,
        selected_known_original,
        train_ratio=train_ratio,
        seed=split_seed,
    )

    result: Dict[str, object] = {
        "day1_known_train": {
            "X": x_train,
            "y": y_train,
            "day": "ADS-B known-pool train partition",
            "tx_range": "0-89",
            "split": f"stratified {train_ratio:.0%}",
        },
        "day1_initial_eval": {
            "X": x_known_eval,
            "y": y_known_eval,
            "day": "ADS-B known-pool held-out partition",
            "tx_range": "0-89",
            "split": f"stratified {1.0 - train_ratio:.0%}",
        },
    }

    cumulative_unknown_x: List[np.ndarray] = []
    cumulative_unknown_y: List[np.ndarray] = []
    round_counts = []
    for round_index in range(1, num_rounds + 1):
        local_start = (round_index - 1) * round_size
        local_end = round_index * round_size
        local_classes = range(local_start, local_end)
        x_discovery, y_discovery = _select_unknown_group(
            x_unknown_train,
            y_unknown_train,
            local_classes,
            global_offset=initial_known_classes,
        )
        x_round_eval, y_round_eval = _select_unknown_group(
            x_unknown_test,
            y_unknown_test,
            local_classes,
            global_offset=initial_known_classes,
        )
        cumulative_unknown_x.append(x_round_eval)
        cumulative_unknown_y.append(y_round_eval)
        x_eval = np.concatenate([x_known_eval, *cumulative_unknown_x], axis=0)
        y_eval = np.concatenate([y_known_eval, *cumulative_unknown_y], axis=0)

        result[f"day{round_index + 1}_unknown_round{round_index}"] = {
            "X": x_discovery,
            "y": y_discovery,
            "day": f"ADS-B novel pool group {local_start}-{local_end - 1}",
            "tx_range": f"{initial_known_classes + local_start}-{initial_known_classes + local_end - 1}",
            "split": "provided unknown training file",
        }
        result[f"day{round_index + 1}_eval_after_r{round_index}"] = {
            "X": x_eval,
            "y": y_eval,
            "day": f"ADS-B held-out evaluation after round {round_index}",
            "tx_range": f"0-{initial_known_classes + local_end - 1}",
            "split": "known held-out + provided unknown test file",
        }
        round_counts.append(
            {
                "round": int(round_index),
                "discovery_samples": int(len(y_discovery)),
                "evaluation_new_samples": int(len(y_round_eval)),
            }
        )

    result.update(
        {
            "dataset": "ADS-B 90 base identities + 30 disjoint novel identities",
            "data_root": str(root),
            "selected_known_original_labels": selected_known_original,
            "known_label_map": {
                int(original): int(global_label)
                for global_label, original in enumerate(selected_known_original)
            },
            "unknown_global_offset": int(initial_known_classes),
            "known_split_audit": known_audit,
            "round_counts": round_counts,
            "normalization": "per-instance complex RMS",
            "sample_length": int(x_train.shape[-1]),
        }
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=DEFAULT_ADSB_ROOT)
    parser.add_argument("--known_selection_seed", type=int, default=2026)
    parser.add_argument("--split_seed", type=int, default=2026)
    args = parser.parse_args()
    splits = load_adsb_90known_3round(
        args.data_root,
        known_selection_seed=args.known_selection_seed,
        split_seed=args.split_seed,
    )
    print("selected known original labels:", splits["selected_known_original_labels"])
    for name, item in splits.items():
        if isinstance(item, dict) and "X" in item:
            print(name, item["X"].shape, np.unique(item["y"]).tolist())
