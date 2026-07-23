# tools/inspect_orbit_rf.py
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import numpy as np

DATA_ROOT = r"C:\Users\123\Downloads\UCLA_RF\orbit_rf_identification_dataset_updated"

# 让 pickle 能找到同文件夹里的 definitions.py / data_loader.py
sys.path.insert(0, DATA_ROOT)


def describe(obj, name="root", depth=0, max_depth=4):
    indent = "  " * depth
    print(f"{indent}{name}: type={type(obj)}")

    if isinstance(obj, np.ndarray):
        print(f"{indent}  shape={obj.shape}, dtype={obj.dtype}")
        if obj.size > 0:
            try:
                print(f"{indent}  min={np.nanmin(obj)}, max={np.nanmax(obj)}")
            except Exception:
                pass
        return

    if isinstance(obj, dict):
        print(f"{indent}  keys={list(obj.keys())}")
        if depth >= max_depth:
            return
        for k, v in obj.items():
            describe(v, name=f"key[{k}]", depth=depth + 1, max_depth=max_depth)
        return

    if isinstance(obj, (list, tuple)):
        print(f"{indent}  len={len(obj)}")
        if len(obj) == 0 or depth >= max_depth:
            return
        for i in range(min(len(obj), 5)):
            describe(obj[i], name=f"{name}[{i}]", depth=depth + 1, max_depth=max_depth)
        return

    # 普通对象，打印属性
    if hasattr(obj, "__dict__"):
        attrs = list(obj.__dict__.keys())
        print(f"{indent}  attrs={attrs}")
        if depth >= max_depth:
            return
        for a in attrs[:10]:
            try:
                describe(getattr(obj, a), name=f"attr[{a}]", depth=depth + 1, max_depth=max_depth)
            except Exception as e:
                print(f"{indent}  failed attr {a}: {repr(e)}")
        return

    try:
        print(f"{indent}  value={str(obj)[:200]}")
    except Exception:
        pass


def load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")


def main():
    print("=" * 100)
    print("DATA_ROOT:", DATA_ROOT)
    print("=" * 100)

    files = [
        "grid_2019_12_25.pkl",
        "grid_2020_02_03.pkl",
        "grid_2020_02_04.pkl",
        "grid_2020_02_05.pkl",
        "grid_2020_02_06.pkl",
    ]

    for fn in files:
        path = os.path.join(DATA_ROOT, fn)
        print("\n" + "=" * 100)
        print("Inspecting:", path)
        print("=" * 100)

        if not os.path.exists(path):
            print("File not found.")
            continue

        data = load_pickle(path)
        describe(data, name=fn, max_depth=4)


if __name__ == "__main__":
    main()