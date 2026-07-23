# tools/check_wisig_envs.py
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

PKL_PATH = r"C:\Users\123\Downloads\ManyTx.pkl\ManyTx.pkl"
SAVE_CSV = "results/wisig_env_stats.csv"

def main():
    with open(PKL_PATH, "rb") as f:
        raw = pickle.load(f)

    data = raw["data"]
    tx_list = raw.get("tx_list", [])
    rx_list = raw.get("rx_list", [])
    date_list = raw.get("capture_date_list", [])
    eq_list = raw.get("equalized_list", [])

    rows = []

    for rx_idx in range(len(rx_list)):
        for date_idx in range(len(date_list)):
            for eq_idx in range(len(eq_list)):
                total_samples = 0
                valid_tx = 0
                empty_tx = 0
                per_tx_counts = []

                for tx_idx in range(len(data)):
                    try:
                        arr = np.asarray(data[tx_idx][rx_idx][date_idx][eq_idx])
                        n = arr.shape[0]
                    except Exception:
                        n = 0

                    per_tx_counts.append(n)
                    total_samples += n

                    if n > 0:
                        valid_tx += 1
                    else:
                        empty_tx += 1

                rows.append({
                    "rx_idx": rx_idx,
                    "rx_name": rx_list[rx_idx],
                    "date_idx": date_idx,
                    "date": date_list[date_idx],
                    "eq_idx": eq_idx,
                    "eq": eq_list[eq_idx],
                    "total_samples": total_samples,
                    "valid_tx": valid_tx,
                    "empty_tx": empty_tx,
                    "min_samples_per_tx": int(np.min(per_tx_counts)),
                    "max_samples_per_tx": int(np.max(per_tx_counts)),
                    "mean_samples_per_tx": float(np.mean(per_tx_counts)),
                })

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["valid_tx", "total_samples"],
        ascending=[False, False],
    )

    Path("results").mkdir(exist_ok=True)
    df.to_csv(SAVE_CSV, index=False, encoding="utf-8-sig")

    print("\n========== WiSig Environment Statistics ==========")
    print(df.to_string(index=False))
    print(f"\nSaved CSV: {SAVE_CSV}")

    print("\n========== Recommended usable environments ==========")
    usable = df[df["valid_tx"] >= 120].copy()
    print(usable.head(20).to_string(index=False))

if __name__ == "__main__":
    main()