"""汇总 Stage81 LoRa transmission-prototype GPCC seed7 结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# 与 Stage80/Stage46 使用同一个 raw-s28 seed7 对照，保证只比较 discovery
# 前端变化，不把数据窗口、训练轮数和后端变化混进结论。
BASELINE = {
    "overall": 0.3005,
    "old": 0.2429,
    "new": 0.5310,
    "forgetting": 0.1155,
}


def _last_r3(frame: pd.DataFrame) -> pd.Series:
    """兼容 incremental_results 与 clustering_results 的最后一轮表头。"""

    if "Stage" in frame.columns:
        rows = frame[frame["Stage"].astype(str).str.upper() == "AFTER R3"]
    else:
        rows = frame[frame["Round"].astype(str).str.upper() == "R3"]
    return rows.iloc[-1] if not rows.empty else frame.iloc[-1]


def main() -> int:
    """读取结果 CSV 并生成 Markdown/JSON 报告。"""

    parser = argparse.ArgumentParser(description="Stage81 LoRa transmission-prototype GPCC 报告器")
    parser.add_argument("--root", default="results/stage81")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    save_dir = root / f"lora_transmission_prototype_seed7_{args.job_id}"
    incremental = _last_r3(pd.read_csv(save_dir / "incremental_results.csv"))
    clustering = _last_r3(pd.read_csv(save_dir / "clustering_results.csv"))
    metrics = {
        "overall": float(incremental["Overall Acc"]),
        "old": float(incremental["Old Acc"]),
        "new": float(incremental["New Acc"]),
        "forgetting": float(incremental["Forgetting Rate"]),
        "macro_f1": float(incremental.get("Macro F1", 0.0)),
        "hungarian": float(clustering["Hungarian Acc"]),
        "ari": float(clustering["ARI"]),
        "purity": float(clustering["Purity"]),
        "final_clusters": int(clustering["Final Cluster Count"]),
    }
    deltas = {
        key: metrics[key] - BASELINE[key]
        for key in ("overall", "old", "new", "forgetting")
    }
    passed = (
        deltas["overall"] > 0.0
        and deltas["old"] >= -0.02
        and deltas["new"] >= -0.02
    )

    lines = [
        f"# Stage81 LoRa transmission-prototype GPCC seed7 报告（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'通过 seed7 门槛，可继续三种子。' if passed else '未通过 seed7 门槛，暂不扩三种子。'}",
        "- 新前端：先按 transmission 聚合 mean/median/std，再运行 GPCC，最后广播回 symbol。",
        "- 协议边界：只使用可观测 recording_id 和当前 discovery 特征，不读取 held-out eval 真值。",
        "",
        "## R3 指标",
        "",
        "| Variant | Overall | Old | New | Forgetting | Macro F1 | Hungarian | ARI | Purity | Final Clusters |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| stage46_raw_s28_seed7 | {BASELINE['overall']:.4f} | {BASELINE['old']:.4f} | "
            f"{BASELINE['new']:.4f} | {BASELINE['forgetting']:.4f} | n/a | n/a | n/a | n/a | 5 |"
        ),
        (
            f"| stage81_transmission_prototype | {metrics['overall']:.4f} | {metrics['old']:.4f} | "
            f"{metrics['new']:.4f} | {metrics['forgetting']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{metrics['hungarian']:.4f} | {metrics['ari']:.4f} | {metrics['purity']:.4f} | "
            f"{metrics['final_clusters']} |"
        ),
        "",
        "## Delta vs Stage46 Seed7",
        "",
        f"- Overall: {deltas['overall']:+.4f}",
        f"- Old: {deltas['old']:+.4f}",
        f"- New: {deltas['new']:+.4f}",
        f"- Forgetting: {deltas['forgetting']:+.4f}",
        "",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage81_lora_transmission_prototype_seed7_summary_v1",
                "job_id": str(args.job_id),
                "baseline": BASELINE,
                "metrics": metrics,
                "deltas": deltas,
                "passed_seed7_gate": bool(passed),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Stage81 报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
