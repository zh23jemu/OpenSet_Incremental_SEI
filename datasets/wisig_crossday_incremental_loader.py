import pickle
import numpy as np


def load_wisig_crossday_incremental(
    dataset_path=r"D:\WiSigCustom\WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl",
    transpose_to_model=True,
    train_ratio=0.70,
    eq_index=0,
):
    """
    WiSig Cross-Day 10-known + 3-round 7:3 incremental protocol.

    Day1 Tx0-9:
        70% train initial closed-set model
        30% evaluate initial model
    Day2 Tx10-19:
        70% discover/enroll R1 unknown classes
        30% evaluate Tx0-19
    Day3 Tx20-29:
        70% discover/enroll R2 unknown classes
        30% evaluate Tx0-29
    Day4 Tx30-39:
        70% discover/enroll R3 unknown classes
        30% evaluate Tx0-39

    Expected dataset structure:
        data[tx_i][rx_i][day_i][eq_index] -> (N, 256, 2)
    """
    with open(dataset_path, "rb") as f:
        ds = pickle.load(f)

    data = ds["data"]
    tx_list = ds["tx_list"]
    rx_list = ds["rx_list"]
    day_list = ds["capture_date_list"]

    if len(tx_list) < 40 or len(day_list) < 4:
        raise ValueError(f"Expected at least 40 Tx and 4 days, got {len(tx_list)} Tx and {len(day_list)} days")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")

    def collect(tx_indices, day_i, part):
        X_list, y_list = [], []

        for tx_i in tx_indices:
            for rx_i in range(len(rx_list)):
                arr = np.asarray(data[tx_i][rx_i][day_i][eq_index])  # (300, 256, 2)
                split_idx = int(arr.shape[0] * train_ratio)

                if part == "train":
                    arr = arr[:split_idx]
                elif part == "eval":
                    arr = arr[split_idx:]
                else:
                    raise ValueError("part must be 'train' or 'eval'")

                if transpose_to_model:
                    arr = np.transpose(arr, (0, 2, 1))  # (N, 256, 2) -> (N, 2, 256)

                X_list.append(arr.astype(np.float32))
                y_list.append(np.full((arr.shape[0],), tx_i, dtype=np.int64))

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        return X.astype(np.float32), y.astype(np.int64)

    known_tx = list(range(0, 10))
    r1_tx = list(range(10, 20))
    r2_tx = list(range(20, 30))
    r3_tx = list(range(30, 40))

    X_day1_train, y_day1_train = collect(known_tx, day_i=0, part="train")
    X_day1_eval, y_day1_eval = collect(known_tx, day_i=0, part="eval")

    X_r1_discovery, y_r1_discovery = collect(r1_tx, day_i=1, part="train")
    X_r1_eval, y_r1_eval = collect(known_tx + r1_tx, day_i=1, part="eval")

    X_r2_discovery, y_r2_discovery = collect(r2_tx, day_i=2, part="train")
    X_r2_eval, y_r2_eval = collect(known_tx + r1_tx + r2_tx, day_i=2, part="eval")

    X_r3_discovery, y_r3_discovery = collect(r3_tx, day_i=3, part="train")
    X_r3_eval, y_r3_eval = collect(known_tx + r1_tx + r2_tx + r3_tx, day_i=3, part="eval")

    return {
        "day1_known_train": {"X": X_day1_train, "y": y_day1_train, "known_classes": known_tx, "tx_names": tx_list[0:10], "day": day_list[0], "split": "70%"},
        "day1_initial_eval": {"X": X_day1_eval, "y": y_day1_eval, "eval_classes": known_tx, "tx_names": tx_list[0:10], "day": day_list[0], "split": "30%"},
        "day2_unknown_round1": {"X": X_r1_discovery, "y": y_r1_discovery, "unknown_classes": r1_tx, "tx_names": tx_list[10:20], "day": day_list[1], "split": "70%"},
        "day2_eval_after_r1": {"X": X_r1_eval, "y": y_r1_eval, "eval_classes": known_tx + r1_tx, "tx_names": tx_list[0:20], "day": day_list[1], "split": "30%"},
        "day3_unknown_round2": {"X": X_r2_discovery, "y": y_r2_discovery, "unknown_classes": r2_tx, "tx_names": tx_list[20:30], "day": day_list[2], "split": "70%"},
        "day3_eval_after_r2": {"X": X_r2_eval, "y": y_r2_eval, "eval_classes": known_tx + r1_tx + r2_tx, "tx_names": tx_list[0:30], "day": day_list[2], "split": "30%"},
        "day4_unknown_round3": {"X": X_r3_discovery, "y": y_r3_discovery, "unknown_classes": r3_tx, "tx_names": tx_list[30:40], "day": day_list[3], "split": "70%"},
        "day4_eval_after_r3": {"X": X_r3_eval, "y": y_r3_eval, "eval_classes": known_tx + r1_tx + r2_tx + r3_tx, "tx_names": tx_list[0:40], "day": day_list[3], "split": "30%"},
        "train_ratio": train_ratio,
    }


if __name__ == "__main__":
    splits = load_wisig_crossday_incremental()
    for name, item in splits.items():
        if isinstance(item, dict) and "X" in item:
            X, y = item["X"], item["y"]
            print(f"{name}: {X.shape}, {y.shape}, classes={np.unique(y).size}, labels={np.min(y)}-{np.max(y)}, day={item.get('day')}, split={item.get('split')}")
