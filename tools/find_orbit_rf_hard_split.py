# tools/find_orbit_rf_hard_split.py
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np

DATA_ROOT = r"C:\Users\123\Downloads\UCLA_RF\orbit_rf_identification_dataset_updated"
TRAIN_FILE = "grid_2019_12_25.pkl"
TEST_FILE = "grid_2020_02_06.pkl"
MIN_SAMPLES = 50
NUM_KNOWN = 10
NUM_UNKNOWN = 10


def load_pkl(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")


def node_counts(raw):
    names = list(raw["node_list"])
    counts = [int(np.asarray(a).shape[0]) for a in raw["data"]]
    return list(zip(names, counts))


def main():
    train_path = os.path.join(DATA_ROOT, TRAIN_FILE)
    test_path = os.path.join(DATA_ROOT, TEST_FILE)

    train = load_pkl(train_path)
    test = load_pkl(test_path)

    train_counts = node_counts(train)
    test_counts = node_counts(test)

    train_ok = [(n, c) for n, c in train_counts if c >= MIN_SAMPLES]
    test_ok = [(n, c) for n, c in test_counts if c >= MIN_SAMPLES]

    train_names = set(n for n, _ in train_ok)
    test_names = set(n for n, _ in test_ok)

    common = sorted(train_names & test_names)
    train_only = sorted(train_names - test_names)
    test_only = sorted(test_names - train_names)

    print("=" * 100)
    print("ORBIT/UCLA RF hard split finder")
    print("=" * 100)
    print("DATA_ROOT:", DATA_ROOT)
    print("TRAIN_FILE:", TRAIN_FILE)
    print("TEST_FILE:", TEST_FILE)
    print("MIN_SAMPLES:", MIN_SAMPLES)
    print()
    print(f"train nodes total: {len(train['node_list'])}, usable >= {MIN_SAMPLES}: {len(train_ok)}")
    print(f"test nodes total:  {len(test['node_list'])}, usable >= {MIN_SAMPLES}: {len(test_ok)}")
    print(f"common usable nodes: {len(common)}")
    print(f"train-only usable nodes: {len(train_only)}")
    print(f"test-only usable nodes: {len(test_only)}")

    # Known: train-side nodes with enough samples.
    known_nodes = [n for n, c in train_ok[:NUM_KNOWN]]

    # Super-hard unknown: prefer test-only nodes, because these are absent from the training-day file.
    unknown_pool = test_only if len(test_only) >= NUM_UNKNOWN else [n for n, c in test_ok if n not in known_nodes]
    unknown_nodes = unknown_pool[:NUM_UNKNOWN]

    print("\nRecommended SUPER-HARD split:")
    print("Known nodes from train file:")
    print(known_nodes)
    print("Unknown nodes from test file:")
    print(unknown_nodes)

    print("\nCopy these arguments:")
    print('--train_file "{}" --test_file "{}"'.format(TRAIN_FILE, TEST_FILE))
    print('--known_node_names "{}"'.format(",".join(known_nodes)))
    print('--unknown_node_names "{}"'.format(",".join(unknown_nodes)))

    print("\nTop train candidates:")
    for n, c in train_ok[:30]:
        print(f"  {n:8s} {c}")

    print("\nTop test-only unknown candidates:")
    test_count_dict = dict(test_counts)
    for n in test_only[:50]:
        print(f"  {n:8s} {test_count_dict[n]}")


if __name__ == "__main__":
    main()
