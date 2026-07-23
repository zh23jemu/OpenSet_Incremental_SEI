# -*- coding: utf-8 -*-
"""
One-command RF feature-set ablation runner.

Before running:
  1) Put rf_features_ablation.py into your project as: features/rf_features.py
  2) Put exp_wisig_multiview_graph_hard_MV_IARC_ablation.py into: experiments/
  3) Run this script from your project root.

Example:
  python run_rf_feature_ablation.py --disable_visualization

Fast default:
  - no training by default; it loads the existing checkpoint
  - visualization disabled
  - only RF-affected methods are run by default: RF only, Graph fusion, MV-IARC
  - output is compact

Outputs:
  results/rf_feature_ablation/
    rf_feature_ablation_long.csv
    rf_feature_ablation_compact.csv
    best_by_method.csv
    <feature_set>/per_round_summary_results.csv
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_FEATURE_SETS = [
    "amp",
    "phase",
    "iq",
    "spectral",
    "amp_phase",
    "amp_iq",
    "phase_iq",
    "iq_spectral",
    "no_amp",
    "no_phase",
    "no_iq",
    "no_spectral",
    "all24",
]


def parse_csv_list(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def safe_name(text):
    out = str(text).strip().lower()
    for a, b in [(" ", "_"), ("/", "_"), ("\\", "_"), ("+", "plus")]:
        out = out.replace(a, b)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def run_one(feature_set, args, unknown_args):
    save_dir = Path(args.base_save_dir) / safe_name(feature_set)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["RF_FEATURE_SET"] = feature_set
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        args.experiment_script,
        "--dataset_path", args.dataset_path,
        "--save_dir", str(save_dir),
        "--checkpoint", args.checkpoint,
        "--methods", args.methods,
        "--compact_output",
        "--disable_visualization",
        "--adaptive_fusion",
        "--alpha_min", str(args.alpha_min),
        "--alpha_max", str(args.alpha_max),
        "--min_cluster_size", str(args.min_cluster_size),
        "--min_samples", str(args.min_samples),
        "--top_k", str(args.top_k),
        "--merge_threshold", str(args.merge_threshold),
        "--reliability_threshold", str(args.reliability_threshold),
        "--iarc_secondary_min_cluster_size", str(args.iarc_secondary_min_cluster_size),
        "--iarc_secondary_min_samples", str(args.iarc_secondary_min_samples),
        "--iarc_new_threshold", str(args.iarc_new_threshold),
        "--iarc_merge_threshold", str(args.iarc_merge_threshold),
    ]

    if args.train_closedset:
        cmd += ["--train_closedset", "--epochs", str(args.epochs)]

    cmd += list(unknown_args)

    log_path = save_dir / "run.log"
    print(f"\n[RF_FEATURE_SET={feature_set}] running...", flush=True)
    print(" ".join(cmd), flush=True)

    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if proc.returncode != 0:
        print(f"[FAILED] {feature_set}. See log: {log_path}", flush=True)
        return None

    summary_path = save_dir / "per_round_summary_results.csv"
    if not summary_path.exists():
        print(f"[FAILED] {feature_set}. Missing {summary_path}", flush=True)
        return None

    df = pd.read_csv(summary_path)
    df.insert(0, "RF Feature Set", feature_set)
    print(f"[OK] {feature_set}", flush=True)
    return df


def summarize(all_rows, output_dir):
    long_df = pd.concat(all_rows, ignore_index=True)

    metric_cols = ["Purity", "NMI", "Overall Acc", "Forgetting Rate"]
    for col in metric_cols:
        long_df[col] = pd.to_numeric(long_df[col], errors="coerce")

    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / "rf_feature_ablation_long.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

    # Compact method-level summary: enough to see who helps and who hurts.
    rows = []
    for (fs, method), g in long_df.groupby(["RF Feature Set", "Method"], sort=False):
        r3 = g[g["Round"].astype(str) == "R3"]
        rows.append({
            "RF Feature Set": fs,
            "Method": method,
            "Avg Overall Acc": g["Overall Acc"].mean(),
            "R1 Acc": float(g.loc[g["Round"].astype(str) == "R1", "Overall Acc"].mean()),
            "R2 Acc": float(g.loc[g["Round"].astype(str) == "R2", "Overall Acc"].mean()),
            "R3 Acc": float(r3["Overall Acc"].mean()) if len(r3) else float("nan"),
            "Avg NMI": g["NMI"].mean(),
            "Avg Purity": g["Purity"].mean(),
            "R3 Forgetting": float(r3["Forgetting Rate"].mean()) if len(r3) else float("nan"),
            "Avg Forgetting": g["Forgetting Rate"].mean(),
        })

    compact = pd.DataFrame(rows)
    compact = compact.sort_values(["Method", "Avg Overall Acc"], ascending=[True, False])
    compact_path = output_dir / "rf_feature_ablation_compact.csv"
    compact.to_csv(compact_path, index=False, encoding="utf-8-sig")

    best = compact.sort_values("Avg Overall Acc", ascending=False).groupby("Method", as_index=False).head(3)
    best_path = output_dir / "best_by_method.csv"
    best.to_csv(best_path, index=False, encoding="utf-8-sig")

    print("\n========== RF Feature Ablation: Top Results by Method ==========")
    with pd.option_context("display.max_columns", 50, "display.width", 220):
        print(best.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nSaved:")
    print(f"  {long_path}")
    print(f"  {compact_path}")
    print(f"  {best_path}")


def main():
    parser = argparse.ArgumentParser(description="Run RF feature-set ablation in one command.")
    parser.add_argument("--experiment_script", type=str, default="experiments/exp_wisig_multiview_graph_hard_MV_IARC_ablation.py")
    parser.add_argument("--dataset_path", type=str, default=r"D:\WiSigCustom\WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl")
    parser.add_argument("--checkpoint", type=str, default=r"./results/wisig_crossday_10known_3round_7_3/closedset_day1_10known.pth")
    parser.add_argument("--base_save_dir", type=str, default=r"./results/rf_feature_ablation")

    parser.add_argument("--feature_sets", type=str, default=",".join(DEFAULT_FEATURE_SETS), help="Comma-separated RF feature sets to evaluate.")
    parser.add_argument("--methods", type=str, default="RF only,Graph fusion,MV-IARC", help="Comma-separated methods. Deep only is skipped by default because it is not affected by RF features.")

    parser.add_argument("--train_closedset", action="store_true", help="Train closed-set model in each run. Usually keep this OFF and reuse checkpoint for speed.")
    parser.add_argument("--epochs", type=int, default=20)

    # Defaults matching your current experiment, kept here so one command is enough.
    parser.add_argument("--alpha_min", type=float, default=0.65)
    parser.add_argument("--alpha_max", type=float, default=0.95)
    parser.add_argument("--min_cluster_size", type=int, default=100)
    parser.add_argument("--min_samples", type=int, default=15)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--merge_threshold", type=float, default=0.8)
    parser.add_argument("--reliability_threshold", type=float, default=0.3)
    parser.add_argument("--iarc_secondary_min_cluster_size", type=int, default=50)
    parser.add_argument("--iarc_secondary_min_samples", type=int, default=5)
    parser.add_argument("--iarc_new_threshold", type=float, default=0.35)
    parser.add_argument("--iarc_merge_threshold", type=float, default=0.78)

    args, unknown_args = parser.parse_known_args()

    feature_sets = parse_csv_list(args.feature_sets)
    output_dir = Path(args.base_save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("========== RF Feature Ablation Runner ==========")
    print(f"Feature sets: {feature_sets}")
    print(f"Methods: {args.methods}")
    print(f"Output dir: {output_dir}")
    print("Visualization: disabled")
    print("Training closed-set:", "ON" if args.train_closedset else "OFF, using checkpoint")

    all_rows = []
    failed = []
    for fs in feature_sets:
        df = run_one(fs, args, unknown_args)
        if df is None:
            failed.append(fs)
        else:
            all_rows.append(df)

    if not all_rows:
        raise RuntimeError(f"All ablation runs failed. Failed feature sets: {failed}")

    summarize(all_rows, output_dir)

    if failed:
        print("\nFailed feature sets:", failed)
        print("Check each failed run.log under the corresponding feature-set folder.")


if __name__ == "__main__":
    main()
