import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class WiSigManyTxDataset(Dataset):
    """
    WiSig ManyTx dataset loader.

    Expected data index:
        raw["data"][tx_idx][rx_idx][date_idx][eq_idx]

    One signal original shape:
        [L, 2]

    Model input shape:
        [2, L]
    """

    def __init__(
        self,
        pkl_path: str,
        selected_tx_indices,
        tx_label_map,
        rx_idx: int = 0,
        date_idx: int = 3,
        eq_idx: int = 0,
        max_samples_per_tx: int = 50,
    ):
        super().__init__()

        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)

        self.raw = raw
        self.tx_list = raw["tx_list"]
        self.rx_list = raw["rx_list"]
        self.capture_date_list = raw["capture_date_list"]
        self.equalized_list = raw["equalized_list"]

        self.selected_tx_indices = selected_tx_indices
        self.tx_label_map = tx_label_map
        self.rx_idx = rx_idx
        self.date_idx = date_idx
        self.eq_idx = eq_idx
        self.max_samples_per_tx = max_samples_per_tx

        self.samples = []
        self.labels = []
        self.tx_indices = []

        for tx_idx in selected_tx_indices:
            if tx_idx not in tx_label_map:
                raise ValueError(f"tx_idx {tx_idx} not found in tx_label_map.")

            label = tx_label_map[tx_idx]

            try:
                sigs = raw["data"][tx_idx][rx_idx][date_idx][eq_idx]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load data[tx={tx_idx}][rx={rx_idx}]"
                    f"[date={date_idx}][eq={eq_idx}]."
                ) from e

            sigs = np.asarray(sigs, dtype=np.float32)

            if sigs.ndim != 3:
                raise ValueError(
                    f"Expected signals shape [N, L, 2], "
                    f"but got {sigs.shape} for tx_idx={tx_idx}"
                )

            if sigs.shape[-1] != 2:
                raise ValueError(
                    f"Expected last dimension to be I/Q=2, "
                    f"but got {sigs.shape} for tx_idx={tx_idx}"
                )

            if max_samples_per_tx is not None:
                sigs = sigs[:max_samples_per_tx]

            for s in sigs:
                # [L, 2] -> [2, L]
                x = s.T.astype(np.float32)

                self.samples.append(x)
                self.labels.append(label)
                self.tx_indices.append(tx_idx)

        if len(self.samples) == 0:
            raise RuntimeError("No samples loaded. Please check selected Tx/Rx/date/eq settings.")

        self.samples = np.stack(self.samples, axis=0).astype(np.float32)
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.tx_indices = np.asarray(self.tx_indices, dtype=np.int64)

        print("=" * 80)
        print("Loaded WiSig ManyTx Dataset")
        print(f"pkl_path: {pkl_path}")
        print(f"selected_tx count: {len(selected_tx_indices)}")
        print(f"selected_tx_indices: {selected_tx_indices}")
        print(f"rx_idx: {rx_idx}, rx_name: {self.rx_list[rx_idx]}")
        print(f"date_idx: {date_idx}, date: {self.capture_date_list[date_idx]}")
        print(f"eq_idx: {eq_idx}, equalized: {self.equalized_list[eq_idx]}")
        print(f"samples shape: {self.samples.shape}")
        print(f"labels shape: {self.labels.shape}")
        print("=" * 80)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.samples[idx])  # [2, L]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        tx_idx = torch.tensor(self.tx_indices[idx], dtype=torch.long)
        return x, y, tx_idx


def build_manytx_known_unknown_datasets(
    pkl_path: str,
    num_known: int = 20,
    num_unknown: int = 10,
    rx_idx: int = 0,
    date_idx: int = 3,
    eq_idx: int = 0,
    max_samples_per_tx: int = 50,
):
    """
    Build ManyTx known training dataset and all-test dataset.

    Example:
        num_known = 20
        num_unknown = 10

        known Tx:
            0, 1, ..., 19

        unknown Tx:
            20, 21, ..., 29

        all_test:
            0, 1, ..., 29

    Returns:
        known_dataset:
            only known Tx, labels are 0 to num_known-1

        all_test_dataset:
            known + unknown Tx, labels are 0 to num_known+num_unknown-1

        known_tx_indices:
            original Tx indices for known classes

        unknown_tx_indices:
            original Tx indices for unknown classes
    """

    known_tx_indices = list(range(num_known))
    unknown_tx_indices = list(range(num_known, num_known + num_unknown))
    all_tx_indices = known_tx_indices + unknown_tx_indices

    known_label_map = {
        tx_idx: label for label, tx_idx in enumerate(known_tx_indices)
    }

    all_label_map = {
        tx_idx: label for label, tx_idx in enumerate(all_tx_indices)
    }

    known_dataset = WiSigManyTxDataset(
        pkl_path=pkl_path,
        selected_tx_indices=known_tx_indices,
        tx_label_map=known_label_map,
        rx_idx=rx_idx,
        date_idx=date_idx,
        eq_idx=eq_idx,
        max_samples_per_tx=max_samples_per_tx,
    )

    all_test_dataset = WiSigManyTxDataset(
        pkl_path=pkl_path,
        selected_tx_indices=all_tx_indices,
        tx_label_map=all_label_map,
        rx_idx=rx_idx,
        date_idx=date_idx,
        eq_idx=eq_idx,
        max_samples_per_tx=max_samples_per_tx,
    )

    return known_dataset, all_test_dataset, known_tx_indices, unknown_tx_indices


def inspect_manytx_pkl(pkl_path: str):
    """
    Utility function to inspect ManyTx.pkl structure.
    """

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print("=" * 80)
    print(f"Inspecting: {pkl_path}")
    print("=" * 80)

    print("Top-level type:", type(data))

    print("\nKeys:")
    for k in data.keys():
        print(" ", k)

    print("\nBasic information:")
    print("tx_list length:", len(data["tx_list"]))
    print("rx_list length:", len(data["rx_list"]))
    print("capture_date_list length:", len(data["capture_date_list"]))
    print("equalized_list length:", len(data["equalized_list"]))
    print("max_sig:", data["max_sig"])

    print("\nFirst few values:")
    print("tx_list first 10:", data["tx_list"][:10])
    print("rx_list first 10:", data["rx_list"][:10])
    print("capture_date_list:", data["capture_date_list"])
    print("equalized_list:", data["equalized_list"])

    print("\nNested data structure check:")
    d = data["data"]

    print("len(data) =", len(d))
    print("len(data[0]) =", len(d[0]))
    print("len(data[0][0]) =", len(d[0][0]))
    print("len(data[0][0][0]) =", len(d[0][0][0]))

    sample = d[0][0][0][0]
    sample_np = np.asarray(sample)

    print("type(sample) =", type(sample))
    print("sample shape =", sample_np.shape)

    if len(sample_np) > 0:
        print("one signal shape =", np.asarray(sample_np[0]).shape)

    print("=" * 80)