"""汇总 Stage82 LoRa transmission-symbol pooling seed7 结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASELINE = {
    "overall": 0.3005,
    "old": 0.2429,
    "new": 0.5310,
    "forgetting": 0.1155,
}


def _r3_row(path: Path) -> pd.Series:
    """读取 R3 行，兼容主实验的 Stage/Round 两种表头。"""
    frame = pd.read_csv(path)
    if "Stage" in frame.columns:
        values = frame["Stage"].astype(str).str.strip().str.lower()
        rows = frame.loc[values.isin({"after r3", "r3"})]
    elif "Round" in frame.columns:
        values = frame["Round"].astype(str).str.strip().str.lower()
        rows = frame.loc[values.isin({"r3", "3"})]
    else:
        rows = frame.iloc[[-1]]
    return rows.iloc[-1] if not rows.empty else frame.iloc[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage82 LoRa transmission pooling 报告器")
    parser.add_argument("--root", default="results/stage82")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--mode", choices=["discovery", "cil"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    prefix = "lora_tx_pool_discovery_seed7" if args.mode == "discovery" else "lora_tx_pool_cil_seed7"
    save_dir = root / f"{prefix}_{args.job_id}"
    clustering = _r3_row(save_dir / "clustering_results.csv")
    metrics = {
        "hungarian": float(clustering["Hungarian Acc"]),
        "ari": float(clustering["ARI"]),
        "purity": float(clustering["Purity"]),
        "final_clusters": int(clustering["Final Cluster Count"]),
    }
    if args.mode == "cil":
        incremental = _r3_row(save_dir / "incremental_results.csv")
        metrics.update(
            {
                "overall": float(incremental["Overall Acc"]),
                "old": float(incremental["Old Acc"]),
                "new": float(incremental["New Acc"]),
                "forgetting": float(incremental["Forgetting Rate"]),
                "macro_f1": float(incremental.get("Macro F1", 0.0)),
            }
        )
        deltas = {
            key: metrics[key] - BASELINE[key]
            for key in ("overall", "old", "new", "forgetting")
        }
        passed = (
            deltas["overall"] > 0.0
            and deltas["old"] >= -0.02
            and deltas["new"] >= -0.02
        )
    else:
        deltas = {}
        passed = (
            metrics["final_clusters"] == 5
            and metrics["hungarian"] > 0.4177
            and metrics["ari"] >= 0.1471
        )

    lines = [
        f"# Stage82 LoRa transmission-symbol pooling {args.mode} seed7 报告（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'通过当前门槛。' if passed else '未通过当前门槛，暂不扩展。'}",
        "- 正式主指标为严格 symbol-level；recording-level 不参与门槛判定。",
        "- transmission pooling 只使用当前 discovery 的可观测 recording_id 和特征，未读取 held-out eval 真值。",
        "",
        "## R3 结果",
        "",
        f"- Clustering Hungarian / ARI / Purity = {metrics['hungarian']:.4f} / {metrics['ari']:.4f} / {metrics['purity']:.4f}",
        f"- Final clusters = {metrics['final_clusters']}",
    ]
    if args.mode == "cil":
        lines.extend(
            [
                f"- Symbol Overall / Old / New / Forgetting = {metrics['overall']:.4f} / {metrics['old']:.4f} / {metrics['new']:.4f} / {metrics['forgetting']:.4f}",
                f"- vs Stage46 seed7 Overall / Old / New / Forgetting = {deltas['overall']:+.4f} / {deltas['old']:+.4f} / {deltas['new']:+.4f} / {deltas['forgetting']:+.4f}",
            ]
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage82_lora_transmission_pooling_summary_v1",
                "job_id": str(args.job_id),
                "mode": args.mode,
                "baseline": BASELINE,
                "metrics": metrics,
                "deltas": deltas,
                "passed_gate": bool(passed),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Stage82 报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
