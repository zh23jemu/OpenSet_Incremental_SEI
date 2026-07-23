# datasets/adsb_loader.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 你的 ADS-B 数据集路径
# ============================================================

DEFAULT_ADSB_ROOT = r"C:\Users\123\Downloads\Dataset\Dataset"


def _to_int_label(y):
    """
    Convert labels to 1D integer labels.

    Supported:
        [N]
        [N, 1]
        one-hot [N, C]
    """
    y = np.asarray(y)

    if y.ndim == 1:
        return y.astype(np.int64)

    if y.ndim == 2:
        if y.shape[1] == 1:
            return y.reshape(-1).astype(np.int64)

        # one-hot label -> integer label
        return np.argmax(y, axis=1).astype(np.int64)

    raise ValueError(f"Unsupported label shape: {y.shape}")


def _fix_iq_shape(x):
    """
    Convert IQ data to [N, 2, L].

    Supported:
        [N, 2, L]
        [N, L, 2]
        [N, L]
    """
    x = np.asarray(x)

    if x.ndim == 2:
        # [N, L] -> [N, 1, L]
        # 这种情况说明数据不是标准 I/Q 双通道
        x = x[:, None, :]
        return x.astype(np.float32)

    if x.ndim == 3:
        # already [N, 2, L]
        if x.shape[1] == 2:
            return x.astype(np.float32)

        # [N, L, 2] -> [N, 2, L]
        if x.shape[-1] == 2:
            x = np.transpose(x, (0, 2, 1))
            return x.astype(np.float32)

    raise ValueError(f"Unsupported IQ shape: {x.shape}")


def normalize_iq_per_sample(x, eps=1e-8):
    """
    Per-sample normalization over time dimension.

    Input:
        x: [N, C, L]

    Output:
        normalized x: [N, C, L]
    """
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    x = (x - mean) / (std + eps)
    return x.astype(np.float32)


def remap_labels(y):
    """
    Remap arbitrary labels to 0, 1, 2, ...

    Example:
        original labels: [90, 91, 92]
        remapped labels: [0, 1, 2]
    """
    unique = sorted(np.unique(y).tolist())

    old_to_new = {old: i for i, old in enumerate(unique)}
    new_to_old = {i: old for old, i in old_to_new.items()}

    y_new = np.array([old_to_new[v] for v in y], dtype=np.int64)

    return y_new, old_to_new, new_to_old


def print_label_distribution(y, max_classes=100):
    """
    Print number of samples for each class.
    """
    unique, counts = np.unique(y, return_counts=True)

    print(f"[ADS-B] Num classes: {len(unique)}")
    print("[ADS-B] Label distribution:")

    for cls, cnt in zip(unique[:max_classes], counts[:max_classes]):
        print(f"  class {cls}: {cnt} samples")

    if len(unique) > max_classes:
        print(f"  ... {len(unique) - max_classes} more classes not shown")


class ADSBDataset(Dataset):
    def __init__(
        self,
        x_path,
        y_path,
        normalize=True,
        label_remap=True,
        verbose=True,
    ):
        super().__init__()

        if not os.path.exists(x_path):
            raise FileNotFoundError(f"X file not found: {x_path}")

        if not os.path.exists(y_path):
            raise FileNotFoundError(f"Y file not found: {y_path}")

        raw_x = np.load(x_path)
        raw_y = np.load(y_path)

        if verbose:
            print("\n========== Loading ADS-B Dataset ==========")
            print(f"[ADS-B] X path: {x_path}")
            print(f"[ADS-B] Y path: {y_path}")
            print(f"[ADS-B] Raw X shape: {raw_x.shape}")
            print(f"[ADS-B] Raw Y shape: {raw_y.shape}")
            print(f"[ADS-B] Raw X dtype: {raw_x.dtype}")
            print(f"[ADS-B] Raw Y dtype: {raw_y.dtype}")

        y = _to_int_label(raw_y)
        x = _fix_iq_shape(raw_x)

        if verbose:
            print(f"[ADS-B] Fixed X shape before norm: {x.shape}")
            print(f"[ADS-B] Fixed Y shape: {y.shape}")

        if normalize:
            x = normalize_iq_per_sample(x)

        if label_remap:
            y, old_to_new, new_to_old = remap_labels(y)
        else:
            old_to_new, new_to_old = None, None

        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long()

        self.old_to_new = old_to_new
        self.new_to_old = new_to_old

        if verbose:
            print(f"[ADS-B] Final X shape: {self.x.shape}")
            print(f"[ADS-B] Final Y shape: {self.y.shape}")
            print_label_distribution(self.y.numpy())
            print("[ADS-B] Load success!")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def build_adsb_paths(data_root=DEFAULT_ADSB_ROOT, test_n_class=30):
    """
    Build ADS-B file paths.

    data_root example:
        C:\\Users\\123\\Downloads\\Dataset\\Dataset

    test_n_class:
        10, 20, or 30
    """
    return {
        "x_train": os.path.join(data_root, "X_train_90Class.npy"),
        "y_train": os.path.join(data_root, "Y_train_90Class.npy"),
        "x_test": os.path.join(data_root, f"X_test_{test_n_class}Class.npy"),
        "y_test": os.path.join(data_root, f"Y_test_{test_n_class}Class.npy"),
    }


def check_file_exists(paths):
    print("\n========== Checking File Paths ==========")

    all_exist = True

    for name, path in paths.items():
        exists = os.path.exists(path)
        print(f"{name}: {path}")
        print(f"  exists: {exists}")

        if not exists:
            all_exist = False

    if not all_exist:
        raise FileNotFoundError(
            "Some ADS-B files are missing. Please check data_root or file names."
        )

    print("[ADS-B] All required files exist.")


def check_adsb_dataset(data_root=DEFAULT_ADSB_ROOT, test_n_class=30, batch_size=64):
    """
    Check whether ADS-B dataset can be loaded correctly.
    """
    paths = build_adsb_paths(data_root, test_n_class)

    check_file_exists(paths)

    print("\n========== Checking Train Set ==========")
    train_set = ADSBDataset(
        x_path=paths["x_train"],
        y_path=paths["y_train"],
        normalize=True,
        label_remap=True,
        verbose=True,
    )

    print("\n========== Checking Test Set ==========")
    test_set = ADSBDataset(
        x_path=paths["x_test"],
        y_path=paths["y_test"],
        normalize=True,
        label_remap=True,
        verbose=True,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    x_train_batch, y_train_batch = next(iter(train_loader))
    x_test_batch, y_test_batch = next(iter(test_loader))

    print("\n========== DataLoader Batch Check ==========")
    print(f"Train batch X shape: {x_train_batch.shape}")
    print(f"Train batch Y shape: {y_train_batch.shape}")
    print(f"Test batch X shape: {x_test_batch.shape}")
    print(f"Test batch Y shape: {y_test_batch.shape}")

    print("\n========== Summary ==========")
    print(f"Train samples: {len(train_set)}")
    print(f"Train classes: {len(torch.unique(train_set.y))}")
    print(f"Test samples: {len(test_set)}")
    print(f"Test classes: {len(torch.unique(test_set.y))}")

    print("\n========== Label Order Check ==========")
    print("First 50 train labels:")
    print(train_set.y[:50].numpy())

    print("First 50 test labels:")
    print(test_set.y[:50].numpy())

    print("\n[ADS-B] Dataset check finished successfully.")

    return train_set, test_set


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_ADSB_ROOT,
        help="Path to ADS-B dataset folder.",
    )

    parser.add_argument(
        "--test_n_class",
        type=int,
        default=30,
        choices=[10, 20, 30],
        help="Which ADS-B test split to check.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    check_adsb_dataset(
        data_root=args.data_root,
        test_n_class=args.test_n_class,
        batch_size=args.batch_size,
    )