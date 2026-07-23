import argparse
import json
import pickle
from pathlib import Path

import numpy as np


STAGES = [
    ("day1_known_train", 0, list(range(0, 4))),
    ("day1_initial_eval", 0, list(range(0, 4))),
    ("day2_unknown_round1", 1, list(range(4, 6))),
    ("day2_eval_after_r1", 1, list(range(0, 6))),
    ("day3_unknown_round2", 2, list(range(6, 8))),
    ("day3_eval_after_r2", 2, list(range(0, 8))),
    ("day4_unknown_round3", 3, list(range(8, 10))),
    ("day4_eval_after_r3", 3, list(range(0, 10))),
]


def to_model_shape(array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (256, 2):
        raise ValueError(f"Expected [N,256,2], got {array.shape}")
    return np.transpose(array, (0, 2, 1)).astype(np.float32)


def pack_stage(raw, tx_indices, rx_index, day_index, equalized_index, sample_slice):
    xs, ys, sample_indices = [], [], []
    for tx_index in tx_indices:
        array = np.asarray(raw["data"][tx_index][rx_index][day_index][equalized_index])
        indices = np.arange(len(array), dtype=np.int16)[sample_slice]
        selected = array[sample_slice]
        if len(selected) == 0:
            raise ValueError(
                f"Empty group: tx={tx_index}, rx={rx_index}, day={day_index}"
            )
        xs.append(to_model_shape(selected))
        ys.append(np.full(len(selected), tx_index, dtype=np.int64))
        sample_indices.append(indices)
    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(sample_indices, axis=0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--discovery_rx", type=int, default=0)
    parser.add_argument("--evaluation_rx", type=int, default=1)
    parser.add_argument("--equalized_index", type=int, default=1)
    parser.add_argument("--fixed_train_ratio", type=float, default=0.70)
    args = parser.parse_args()

    with open(args.pkl_path, "rb") as handle:
        raw = pickle.load(handle)

    if len(raw["data"]) != 10:
        raise ValueError(f"This protocol expects 10 Tx, got {len(raw['data'])}")

    arrays = {}
    manifest_stages = []
    for protocol in ("fixedrx", "crossrx"):
        for stage, day_index, tx_indices in STAGES:
            is_eval = "eval" in stage
            if protocol == "crossrx":
                rx_index = args.evaluation_rx if is_eval else args.discovery_rx
                sample_slice = slice(None)
                split_name = "all samples; disjoint receiver"
            else:
                rx_index = args.discovery_rx
                # Every complete selected group has 200 samples. Split inside
                # each Tx/Rx/day condition so train/discovery and evaluation do
                # not share an IQ instance.
                split_index = int(
                    len(raw["data"][tx_indices[0]][rx_index][day_index][args.equalized_index])
                    * args.fixed_train_ratio
                )
                sample_slice = slice(split_index, None) if is_eval else slice(0, split_index)
                split_name = "last 30%" if is_eval else "first 70%"

            x, y, sample_index = pack_stage(
                raw,
                tx_indices,
                rx_index,
                day_index,
                args.equalized_index,
                sample_slice,
            )
            prefix = f"{protocol}_{stage}"
            arrays[f"{prefix}_X"] = x
            arrays[f"{prefix}_y"] = y
            arrays[f"{prefix}_sample_index"] = sample_index
            arrays[f"{prefix}_rx_index"] = np.full(len(x), rx_index, dtype=np.int16)
            arrays[f"{prefix}_day_index"] = np.full(len(x), day_index, dtype=np.int16)
            manifest_stages.append({
                "protocol": protocol,
                "stage": stage,
                "day_index": day_index,
                "day": raw["capture_date_list"][day_index],
                "tx_indices": tx_indices,
                "rx_index": rx_index,
                "rx": raw["rx_list"][rx_index],
                "sample_split": split_name,
                "samples": int(len(x)),
            })
            print(prefix, x.shape, "classes", np.unique(y).size, flush=True)

    arrays["tx_names"] = np.asarray(raw["tx_list"])
    arrays["rx_names"] = np.asarray(raw["rx_list"])
    arrays["day_names"] = np.asarray(raw["capture_date_list"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)

    manifest = {
        "source": str(Path(args.pkl_path).resolve()),
        "tx_count": len(raw["tx_list"]),
        "rx_count": len(raw["rx_list"]),
        "day_count": len(raw["capture_date_list"]),
        "tx_names": list(raw["tx_list"]),
        "rx_names": list(raw["rx_list"]),
        "day_names": list(raw["capture_date_list"]),
        "equalized_index": args.equalized_index,
        "discovery_rx_index": args.discovery_rx,
        "evaluation_rx_index": args.evaluation_rx,
        "fixed_train_ratio": args.fixed_train_ratio,
        "stages": manifest_stages,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved {output} ({output.stat().st_size} bytes)")
    print(f"saved {manifest_path}")


if __name__ == "__main__":
    main()
