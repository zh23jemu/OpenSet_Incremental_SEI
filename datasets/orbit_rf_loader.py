# datasets/orbit_rf_loader.py
# -*- coding: utf-8 -*-

import os
import pickle
import random
from typing import List, Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset


def load_orbit_pkl(pkl_path: str):
    """Load one ORBIT/UCLA RF grid_*.pkl file."""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"PKL file not found: {pkl_path}")
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(pkl_path, "rb") as f:
            return pickle.load(f, encoding="latin1")


def inspect_orbit_pkl(pkl_path: str, max_items: int = 10):
    raw = load_orbit_pkl(pkl_path)
    print("=" * 100)
    print(f"Inspecting ORBIT/UCLA RF PKL: {pkl_path}")
    print("=" * 100)
    print("Top-level type:", type(raw))
    print("Keys:", list(raw.keys()))
    data = raw["data"]
    node_list = raw["node_list"]
    print("num nodes:", len(node_list))
    print("node_list first values:", node_list[:max_items])
    for i in range(min(max_items, len(data))):
        arr = np.asarray(data[i])
        print(f"  idx={i:3d}, node={node_list[i]}, shape={arr.shape}, dtype={arr.dtype}")
    print("=" * 100)


class OrbitRFDataset(Dataset):
    """
    ORBIT / UCLA RF Identification dataset loader.

    One pkl file structure:
        raw["data"]      : list of arrays, one array per node/transmitter
        raw["node_list"] : list of node IDs, e.g. "13-8."

    One node array shape:
        [N, 256, 2]

    Model input shape:
        [2, 256]
    """

    def __init__(
        self,
        pkl_path: str,
        selected_node_names: Optional[Sequence[str]] = None,
        selected_indices: Optional[Sequence[int]] = None,
        label_map: Optional[Dict[Union[str, int], int]] = None,
        max_samples_per_class: Optional[int] = 100,
        normalize_iq: bool = False,
        random_sample: bool = False,
        seed: int = 7,
    ):
        super().__init__()

        raw = load_orbit_pkl(pkl_path)
        self.pkl_path = pkl_path
        self.raw = raw
        self.node_list = list(raw["node_list"])
        self.data = raw["data"]
        self.normalize_iq = normalize_iq

        if selected_node_names is not None and selected_indices is not None:
            raise ValueError("Use either selected_node_names or selected_indices, not both.")

        if selected_node_names is None and selected_indices is None:
            selected_indices = list(range(len(self.node_list)))

        if selected_node_names is not None:
            node_to_idx = {name: i for i, name in enumerate(self.node_list)}
            missing = [n for n in selected_node_names if n not in node_to_idx]
            if missing:
                raise ValueError(f"Selected node names not found in {pkl_path}: {missing}")
            selected_indices = [node_to_idx[n] for n in selected_node_names]
        else:
            selected_indices = list(selected_indices)
            selected_node_names = [self.node_list[i] for i in selected_indices]

        self.selected_indices = list(selected_indices)
        self.selected_node_names = list(selected_node_names)

        if label_map is None:
            label_map = {name: label for label, name in enumerate(self.selected_node_names)}
        self.label_map = label_map

        rng = np.random.default_rng(seed)

        samples = []
        labels = []
        node_names = []
        original_indices = []

        for idx, node_name in zip(self.selected_indices, self.selected_node_names):
            if node_name in label_map:
                label = label_map[node_name]
            elif idx in label_map:
                label = label_map[idx]
            else:
                raise ValueError(f"No label found for node {node_name} / index {idx}.")

            arr = np.asarray(self.data[idx], dtype=np.float32)
            if arr.ndim != 3 or arr.shape[-1] != 2:
                raise ValueError(
                    f"Expected node array shape [N, 256, 2], got {arr.shape} for node {node_name}."
                )
            if arr.shape[0] == 0:
                raise ValueError(f"Node {node_name} has no samples in {pkl_path}.")

            n = arr.shape[0]
            if max_samples_per_class is not None:
                keep = min(max_samples_per_class, n)
                if random_sample and n > keep:
                    chosen = rng.choice(n, size=keep, replace=False)
                    arr = arr[chosen]
                else:
                    arr = arr[:keep]

            for s in arr:
                # [L, 2] -> [2, L]
                x = s.T.astype(np.float32)

                if normalize_iq:
                    mean = x.mean(axis=1, keepdims=True)
                    std = x.std(axis=1, keepdims=True) + 1e-8
                    x = (x - mean) / std

                samples.append(x)
                labels.append(label)
                node_names.append(node_name)
                original_indices.append(idx)

        if len(samples) == 0:
            raise RuntimeError("No samples loaded. Check selected nodes and max_samples_per_class.")

        self.samples = np.stack(samples, axis=0).astype(np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.node_names = np.asarray(node_names)
        self.original_indices = np.asarray(original_indices, dtype=np.int64)

        print("=" * 100)
        print("Loaded ORBIT/UCLA RF Dataset")
        print(f"pkl_path: {pkl_path}")
        print(f"selected node count: {len(self.selected_node_names)}")
        print(f"selected nodes: {self.selected_node_names}")
        print(f"max_samples_per_class: {max_samples_per_class}")
        print(f"normalize_iq: {normalize_iq}")
        print(f"samples shape: {self.samples.shape}")
        print(f"labels shape: {self.labels.shape}")
        print("=" * 100)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.samples[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
