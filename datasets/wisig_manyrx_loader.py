# datasets/wisig_manyrx_loader.py
# -*- coding: utf-8 -*-

"""
WiSig ManyRx dataset loader.

Expected pickle structure:
    raw["data"][tx_idx][rx_idx][date_idx][eq_idx]

Expected one group shape:
    [N, 256, 2]

Returned model input shape:
    x: [N, 2, 256]
    y: [N]
"""

import pickle
from pathlib import Path
from typing import Dict, Optional, Sequence, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_WISIG_MANYRX_PATH = r"C:\Users\123\Downloads\ManyRx.pkl\ManyRx.pkl"


def _normalize_iq_power(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-sample RMS normalization for IQ data shaped [N, L, 2].
    """
    power = np.mean(np.sum(x ** 2, axis=-1), axis=1, keepdims=True)  # [N, 1]
    scale = np.sqrt(power + eps)[:, None, :]  # [N, 1, 1]
    return x / scale


class WiSigManyRxDataset(Dataset):
    """
    WiSig ManyRx dataset.

    Parameters
    ----------
    pkl_path:
        Path to ManyRx.pkl.

    selected_tx_indices:
        Transmitter indices in ManyRx, e.g. [0,1,2,3,4].

    tx_label_map:
        Dict mapping original tx_idx to class label.
        Example: {0: 0, 1: 1, ..., 4: 4}

    envs:
        List of (rx_idx, date_idx, eq_idx). Each selected environment normally has 200 samples.
        The default 5 environments give 1000 samples per transmitter.

    max_samples_per_tx:
        Maximum number of packets to load per transmitter.

    normalize_iq:
        Whether to apply per-sample RMS normalization.
    """

    def __init__(
        self,
        pkl_path: str = DEFAULT_WISIG_MANYRX_PATH,
        selected_tx_indices: Optional[Sequence[int]] = None,
        tx_label_map: Optional[Dict[int, int]] = None,
        envs: Optional[List[Tuple[int, int, int]]] = None,
        max_samples_per_tx: Optional[int] = 1000,
        normalize_iq: bool = False,
        seed: int = 7,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.pkl_path = Path(pkl_path)

        if not self.pkl_path.exists():
            raise FileNotFoundError(f"WiSig ManyRx.pkl not found: {self.pkl_path}")

        with open(self.pkl_path, "rb") as f:
            raw = pickle.load(f)

        if "data" not in raw:
            raise KeyError("Expected key 'data' in ManyRx.pkl.")

        self.raw = raw
        self.tx_list = raw.get("tx_list", None)
        self.rx_list = raw.get("rx_list", None)
        self.capture_date_list = raw.get("capture_date_list", None)
        self.equalized_list = raw.get("equalized_list", None)

        if selected_tx_indices is None:
            selected_tx_indices = list(range(len(raw["data"])))

        selected_tx_indices = list(selected_tx_indices)

        if tx_label_map is None:
            tx_label_map = {tx_idx: i for i, tx_idx in enumerate(selected_tx_indices)}

        if envs is None:
            envs = [
                (0, 0, 1),
                (0, 1, 1),
                (0, 2, 1),
                (0, 3, 1),
                (1, 0, 1),
            ]

        self.selected_tx_indices = selected_tx_indices
        self.tx_label_map = tx_label_map
        self.envs = list(envs)
        self.max_samples_per_tx = max_samples_per_tx
        self.normalize_iq = normalize_iq
        self.seed = seed
        self.verbose = verbose

        xs = []
        ys = []

        for tx_idx in self.selected_tx_indices:
            if tx_idx < 0 or tx_idx >= len(raw["data"]):
                raise ValueError(
                    f"tx_idx={tx_idx} is out of range for ManyRx. "
                    f"Valid range is 0 to {len(raw['data']) - 1}."
                )

            x_tx = self._load_one_tx(tx_idx)
            label = int(self.tx_label_map[tx_idx])
            y_tx = np.full((x_tx.shape[0],), label, dtype=np.int64)

            xs.append(x_tx)
            ys.append(y_tx)

            if self.verbose:
                tx_name = self.tx_list[tx_idx] if self.tx_list is not None else str(tx_idx)
                print(
                    f"[ManyRx] tx_idx={tx_idx}, tx_name={tx_name}, "
                    f"label={label}, samples={x_tx.shape[0]}"
                )

        x_all = np.concatenate(xs, axis=0).astype(np.float32)
        y_all = np.concatenate(ys, axis=0).astype(np.int64)

        # Convert [N, 256, 2] to [N, 2, 256], same style as ManyTx loader.
        if x_all.ndim == 3 and x_all.shape[-1] == 2:
            x_all = np.transpose(x_all, (0, 2, 1))

        self.x = torch.tensor(x_all, dtype=torch.float32)
        self.y = torch.tensor(y_all, dtype=torch.long)

        if self.verbose:
            print(f"[ManyRx] Final dataset x={self.x.shape}, y={self.y.shape}")

    def _load_one_tx(self, tx_idx: int) -> np.ndarray:
        xs = []

        for rx_idx, date_idx, eq_idx in self.envs:
            try:
                x = np.asarray(self.raw["data"][tx_idx][rx_idx][date_idx][eq_idx])
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load tx_idx={tx_idx}, "
                    f"rx_idx={rx_idx}, date_idx={date_idx}, eq_idx={eq_idx}. "
                    f"Original error: {e}"
                )

            if x.ndim == 3 and x.shape[0] > 0:
                xs.append(x)

        if len(xs) == 0:
            raise ValueError(f"No samples found for tx_idx={tx_idx} in envs={self.envs}")

        x_all = np.concatenate(xs, axis=0).astype(np.float32)

        rng = np.random.default_rng(self.seed + int(tx_idx))
        perm = rng.permutation(x_all.shape[0])
        x_all = x_all[perm]

        if self.max_samples_per_tx is not None:
            x_all = x_all[: int(self.max_samples_per_tx)]

        if self.normalize_iq:
            x_all = _normalize_iq_power(x_all)

        return x_all.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def inspect_manyrx_pkl(pkl_path: str = DEFAULT_WISIG_MANYRX_PATH):
    pkl_path = Path(pkl_path)

    if not pkl_path.exists():
        raise FileNotFoundError(f"WiSig ManyRx.pkl not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    print("Keys:", raw.keys())
    print("TX list:", raw.get("tx_list", None))
    print("RX list:", raw.get("rx_list", None))
    print("Capture date list:", raw.get("capture_date_list", None))
    print("Equalized list:", raw.get("equalized_list", None))
    print("Number of TX:", len(raw["data"]))

    envs = [
        (0, 0, 1),
        (0, 1, 1),
        (0, 2, 1),
        (0, 3, 1),
        (1, 0, 1),
    ]

    print("\nDefault envs:")
    for rx_idx, date_idx, eq_idx in envs:
        rx_name = raw["rx_list"][rx_idx]
        date_name = raw["capture_date_list"][date_idx]
        eq_name = raw["equalized_list"][eq_idx]
        print(f"  rx_idx={rx_idx}, rx={rx_name}, date_idx={date_idx}, date={date_name}, eq_idx={eq_idx}, eq={eq_name}")

    print("\nDefault 1000-sample check:")
    for tx_idx, tx_name in enumerate(raw["tx_list"]):
        xs = []
        for rx_idx, date_idx, eq_idx in envs:
            x = np.asarray(raw["data"][tx_idx][rx_idx][date_idx][eq_idx])
            xs.append(x)
        x_all = np.concatenate(xs, axis=0)
        print(f"  tx_idx={tx_idx}, tx_name={tx_name}, shape={x_all.shape}")


if __name__ == "__main__":
    inspect_manyrx_pkl()

    print("\nBuild test dataset:")
    ds = WiSigManyRxDataset(
        selected_tx_indices=[0, 1, 2, 3, 4],
        tx_label_map={0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        max_samples_per_tx=1000,
        normalize_iq=False,
        seed=7,
    )
    print(ds.x.shape, ds.y.shape)
