import pickle
import numpy as np
from pathlib import Path

pkl_path = Path(r"C:\Users\123\Downloads\ManyRx.pkl\ManyRx.pkl")

with open(pkl_path, "rb") as f:
    raw = pickle.load(f)

# 选择 5 个环境，每个环境大约 200 samples
envs = [
    (0, 0, 1),
    (0, 1, 1),
    (0, 2, 1),
    (0, 3, 1),
    (1, 0, 1),
]

print("Selected environments:")
for rx_idx, date_idx, eq_idx in envs:
    print(
        f"rx_idx={rx_idx}, rx={raw['rx_list'][rx_idx]}, "
        f"date_idx={date_idx}, date={raw['capture_date_list'][date_idx]}, "
        f"eq_idx={eq_idx}, eq={raw['equalized_list'][eq_idx]}"
    )

print("\nChecking 1000 samples per TX...\n")

for tx_idx, tx_name in enumerate(raw["tx_list"]):
    xs = []

    for rx_idx, date_idx, eq_idx in envs:
        x = np.asarray(raw["data"][tx_idx][rx_idx][date_idx][eq_idx])
        xs.append(x)

    x_all = np.concatenate(xs, axis=0)

    print(
        f"tx_idx={tx_idx}, tx_name={tx_name}, "
        f"shape={x_all.shape}"
    )