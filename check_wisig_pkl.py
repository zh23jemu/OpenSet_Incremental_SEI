import pickle
import numpy as np

PKL_PATH = r"D:\RF_Datasets\WiSig\ManyTx\ManyTx.pkl"


def main():
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    print("=" * 80)
    print(f"Loading: {PKL_PATH}")
    print("=" * 80)

    print("Top-level type:", type(data))
    print("Keys:", list(data.keys()))

    print("\nBasic information:")
    print("tx_list length:", len(data["tx_list"]))
    print("rx_list length:", len(data["rx_list"]))
    print("capture_date_list:", data["capture_date_list"])
    print("equalized_list:", data["equalized_list"])
    print("max_sig:", data["max_sig"])

    d = data["data"]

    print("\nNested data structure check:")
    print("len(data) =", len(d))
    print("len(data[0]) =", len(d[0]))
    print("len(data[0][0]) =", len(d[0][0]))
    print("len(data[0][0][0]) =", len(d[0][0][0]))

    sample = np.asarray(d[0][0][0][0])
    print("shape(data[0][0][0][0]) =", sample.shape)

    if sample.shape[0] > 0:
        print("one signal shape =", np.asarray(sample[0]).shape)
    else:
        print("one signal shape = empty sample group")

    print("\nCounting samples for selected setting:")
    rx_idx = 0
    date_idx = 3
    eq_idx = 0

    known_tx = list(range(20))
    unknown_tx = list(range(20, 30))
    selected_tx = known_tx + unknown_tx

    print(f"Selected setting: rx_idx={rx_idx}, date_idx={date_idx}, eq_idx={eq_idx}")

    total = 0
    for tx_idx in selected_tx:
        sigs = np.asarray(d[tx_idx][rx_idx][date_idx][eq_idx])
        n = sigs.shape[0] if sigs.ndim == 3 else 0
        total += n
        print(f"tx_idx={tx_idx:03d}, tx_name={data['tx_list'][tx_idx]}, samples={n}")

    known_total = 0
    for tx_idx in known_tx:
        sigs = np.asarray(d[tx_idx][rx_idx][date_idx][eq_idx])
        known_total += sigs.shape[0] if sigs.ndim == 3 else 0

    unknown_total = 0
    for tx_idx in unknown_tx:
        sigs = np.asarray(d[tx_idx][rx_idx][date_idx][eq_idx])
        unknown_total += sigs.shape[0] if sigs.ndim == 3 else 0

    print("\nKnown / Unknown split:")
    print(f"Known Tx count = {len(known_tx)}, samples = {known_total}")
    print(f"Unknown Tx count = {len(unknown_tx)}, samples = {unknown_total}")
    print(f"All samples = {known_total + unknown_total}")

    print("\nDone.")


if __name__ == "__main__":
    main()

