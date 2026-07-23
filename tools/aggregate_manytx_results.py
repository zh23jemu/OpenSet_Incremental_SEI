import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "Cluster Count",
    "Cluster Count Error",
    "Sample Coverage",
    "Purity",
    "NMI",
    "ARI",
    "Hungarian Acc",
    "Overall Acc",
    "New Acc",
    "Forgetting Rate",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if len(args.runs) != len(args.seeds):
        raise ValueError("--runs and --seeds must have the same length")

    frames = []
    for run, seed in zip(args.runs, args.seeds):
        path = Path(run) / "per_round_summary_results.csv"
        df = pd.read_csv(path)
        df.insert(0, "Seed", seed)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(output_dir / "per_round_all_seeds.csv", index=False, encoding="utf-8-sig")

    grouped = all_df.groupby(["Method", "Round"], sort=False)[METRICS].agg(["mean", "std"])
    grouped.columns = [f"{metric} {stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    grouped.to_csv(output_dir / "per_round_mean_std.csv", index=False, encoding="utf-8-sig")

    method_mean = all_df.groupby("Method", sort=False)[METRICS].mean().reset_index()
    method_mean.to_csv(output_dir / "method_three_round_mean.csv", index=False, encoding="utf-8-sig")
    print(grouped.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
