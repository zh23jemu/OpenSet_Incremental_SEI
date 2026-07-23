# datasets/wisig_manytx_loader.py
# -*- coding: utf-8 -*-

"""
WiSig ManyTx dataset loader.

Expected pickle structure:
    raw["data"][tx_idx][rx_idx][date_idx][eq_idx]

Expected one group shape:
    [N, L, 2]

Returned model input shape:
    x: [2, L]
    y: class label remapped by tx_label_map
"""

import os
import pickle
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_WISIG_MANYTX_PATH = r"C:\Users\123\Downloads\ManyTx.pkl\ManyTx.pkl"


def _normalize_iq_power(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-sample RMS normalization for IQ data shaped [N, L, 2].
    """
    power = np.mean(np.sum(x ** 2, axis=-1), axis=1, keepdims=True)  # [N, 1]
    scale = np.sqrt(power + eps)[:, None, :]  # [N, 1, 1]
    return x / scale


class WiSigManyTxDataset(Dataset):
    """
    WiSig ManyTx dataset.

    Parameters
    ----------
    pkl_path:
        Path to ManyTx.pkl.

    selected_tx_indices:
        Original transmitter indices to load.

    tx_label_map:
        Dict mapping original tx_idx to training/evaluation label.
        Example: {20: 0, 21: 1, ..., 29: 9}

    rx_idx, date_idx, eq_idx:
        Indices for receiver, capture date, and equalization setting.

    max_samples_per_tx:
        Maximum number of packets to load per transmitter. Use None to load all.

    normalize_iq:
        Whether to apply per-sample RMS normalization.
    """

    def __init__(
        self,
        pkl_path: str = DEFAULT_WISIG_MANYTX_PATH,
        selected_tx_indices: Optional[Sequence[int]] = None,
        tx_label_map: Optional[Dict[int, int]] = None,
        rx_idx: int = 0,
        date_idx: int = 0,
        eq_idx: int = 0,
        max_samples_per_tx: Optional[int] = 100,
        normalize_iq: bool = False,
        verbose: bool = True,
    ):
        super().__init__()

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"WiSig ManyTx.pkl not found: {pkl_path}")

        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)

        self.pkl_path = pkl_path
        self.raw = raw
        self.tx_list = raw.get("tx_list", None)
        self.rx_list = raw.get("rx_list", None)
        self.capture_date_list = raw.get("capture_date_list", None)
        self.equalized_list = raw.get("equalized_list", None)
        self.rx_idx = rx_idx
        self.date_idx = date_idx
        self.eq_idx = eq_idx
        self.max_samples_per_tx = max_samples_per_tx
        self.normalize_iq = normalize_iq

        if "data" not in raw:
            raise KeyError("Expected key 'data' in ManyTx.pkl.")

        n_tx_total = len(raw["data"])

        if selected_tx_indices is None:
            selected_tx_indices = list(range(n_tx_total))
        selected_tx_indices = list(selected_tx_indices)

        if tx_label_map is None:
            tx_label_map = {tx_idx: i for i, tx_idx in enumerate(selected_tx_indices)}

        self.selected_tx_indices = selected_tx_indices
        self.tx_label_map = tx_label_map

        samples: List[np.ndarray] = []
        labels: List[int] = []
        tx_indices: List[int] = []

        for tx_idx in selected_tx_indices:
            if tx_idx < 0 or tx_idx >= n_tx_total:
                raise IndexError(f"tx_idx={tx_idx} out of range. Total Tx={n_tx_total}")

            if tx_idx not in tx_label_map:
                raise ValueError(f"tx_idx {tx_idx} is not in tx_label_map.")

            label = int(tx_label_map[tx_idx])

            try:
                sigs = raw["data"][tx_idx][rx_idx][date_idx][eq_idx]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load raw['data'][tx={tx_idx}][rx={rx_idx}]"
                    f"[date={date_idx}][eq={eq_idx}]."
                ) from e

            sigs = np.asarray(sigs, dtype=np.float32)

            if sigs.ndim != 3:
                raise ValueError(
                    f"Expected signals shape [N, L, 2], got {sigs.shape} for tx_idx={tx_idx}."
                )

            if sigs.shape[-1] != 2:
                raise ValueError(
                    f"Expected last dim I/Q=2, got {sigs.shape} for tx_idx={tx_idx}."
                )

            if max_samples_per_tx is not None:
                sigs = sigs[: int(max_samples_per_tx)]

            if normalize_iq:
                sigs = _normalize_iq_power(sigs)

            sigs = np.nan_to_num(sigs, nan=0.0, posinf=0.0, neginf=0.0)

            # [N, L, 2] -> [N, 2, L]
            sigs = np.transpose(sigs, (0, 2, 1)).astype(np.float32)

            samples.append(sigs)
            labels.extend([label] * sigs.shape[0])
            tx_indices.extend([tx_idx] * sigs.shape[0])

        if len(samples) == 0:
            raise RuntimeError("No WiSig samples loaded. Check selected Tx/Rx/date/eq settings.")

        self.samples = np.concatenate(samples, axis=0).astype(np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.tx_indices = np.asarray(tx_indices, dtype=np.int64)

        if verbose:
            self._print_summary()

    def _safe_name(self, seq, idx):
        try:
            return seq[idx]
        except Exception:
            return "unknown"

    def _print_summary(self):
        print("=" * 80)
        print("Loaded WiSig ManyTx Dataset")
        print(f"pkl_path: {self.pkl_path}")
        print(f"selected_tx count: {len(self.selected_tx_indices)}")
        print(f"selected_tx_indices: {self.selected_tx_indices}")
        print(f"rx_idx: {self.rx_idx}, rx_name: {self._safe_name(self.rx_list, self.rx_idx)}")
        print(
            f"date_idx: {self.date_idx}, "
            f"date: {self._safe_name(self.capture_date_list, self.date_idx)}"
        )
        print(f"eq_idx: {self.eq_idx}, equalized: {self._safe_name(self.equalized_list, self.eq_idx)}")
        print(f"normalize_iq: {self.normalize_iq}")
        print(f"samples shape: {self.samples.shape}")
        print(f"labels shape: {self.labels.shape}")
        unique, counts = np.unique(self.labels, return_counts=True)
        print("class distribution:")
        for u, c in zip(unique.tolist(), counts.tolist()):
            print(f"  label {u}: {c} samples")
        print("=" * 80)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.samples[idx])  # [2, L]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def build_manytx_known_unknown_datasets(
    pkl_path: str = DEFAULT_WISIG_MANYTX_PATH,
    num_known: int = 10,
    num_unknown: int = 10,
    train_rx_idx: int = 0,
    train_date_idx: int = 0,
    test_rx_idx: int = 0,
    test_date_idx: int = 0,
    eq_idx: int = 0,
    max_samples_per_tx: Optional[int] = 100,
    normalize_iq: bool = False,
):
    """
    Build known training dataset and unknown test dataset.

    Known Tx:
        0 ... num_known-1

    Unknown Tx:
        num_known ... num_known+num_unknown-1
    """
    known_tx_indices = list(range(num_known))
    unknown_tx_indices = list(range(num_known, num_known + num_unknown))

    known_label_map = {tx_idx: i for i, tx_idx in enumerate(known_tx_indices)}
    unknown_label_map = {tx_idx: i for i, tx_idx in enumerate(unknown_tx_indices)}

    known_dataset = WiSigManyTxDataset(
        pkl_path=pkl_path,
        selected_tx_indices=known_tx_indices,
        tx_label_map=known_label_map,
        rx_idx=train_rx_idx,
        date_idx=train_date_idx,
        eq_idx=eq_idx,
        max_samples_per_tx=max_samples_per_tx,
        normalize_iq=normalize_iq,
    )

    unknown_dataset = WiSigManyTxDataset(
        pkl_path=pkl_path,
        selected_tx_indices=unknown_tx_indices,
        tx_label_map=unknown_label_map,
        rx_idx=test_rx_idx,
        date_idx=test_date_idx,
        eq_idx=eq_idx,
        max_samples_per_tx=max_samples_per_tx,
        normalize_iq=normalize_iq,
    )

    return known_dataset, unknown_dataset, known_tx_indices, unknown_tx_indices


def inspect_manytx_pkl(pkl_path: str = DEFAULT_WISIG_MANYTX_PATH):
    """
    Inspect ManyTx.pkl structure.
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"WiSig ManyTx.pkl not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print("=" * 80)
    print(f"Inspecting: {pkl_path}")
    print("=" * 80)

    print("Top-level type:", type(data))
    print("Keys:", list(data.keys()))

    for key in ["tx_list", "rx_list", "capture_date_list", "equalized_list"]:
        if key in data:
            print(f"{key} length:", len(data[key]))
            print(f"{key} first values:", data[key][:10] if hasattr(data[key], "__getitem__") else data[key])

    if "max_sig" in data:
        print("max_sig:", data["max_sig"])

    d = data["data"]
    print("\nNested structure:")
    print("len(data) =", len(d))
    print("len(data[0]) =", len(d[0]))
    print("len(data[0][0]) =", len(d[0][0]))
    print("len(data[0][0][0]) =", len(d[0][0][0]))

    sample_group = d[0][0][0][0]
    sample_np = np.asarray(sample_group)
    print("sample group type =", type(sample_group))
    print("sample group shape =", sample_np.shape)

    if len(sample_np) > 0:
        one_signal = np.asarray(sample_np[0])
        print("one signal shape =", one_signal.shape)
        print("one signal dtype =", one_signal.dtype)
        try:
            print("one signal min/max =", float(np.nanmin(one_signal)), float(np.nanmax(one_signal)))
        except Exception:
            pass

    print("=" * 80)


if __name__ == "__main__":
    inspect_manytx_pkl(DEFAULT_WISIG_MANYTX_PATH)
