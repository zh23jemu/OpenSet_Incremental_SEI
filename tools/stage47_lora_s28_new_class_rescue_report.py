"""汇总 Stage47 LoRa s28 seed31 新类塌缩修复矩阵。

Stage46 说明完整原始 I/Q 的 s28 多窗配置能提升 Overall、Old 和遗忘率，
但 seed31 的 R3 New Acc 明显塌缩。本报告器只比较 seed31 的小矩阵：
recording-consensus GPCC 与高置信注册筛选是否能拉回新类，同时避免把旧类
保持收益打掉。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASELINE_SEED31 = {
    "overall": 0.2633,
    "old": 0.2554,
    "new": 0.2952,
    "forgetting": 0.2298,
}


def _r3_row(path: Path) -> pd.Series:
    """读取最终 R3 行，兼容 ``Round`` 和 ``Stage=After R3`` 两种 CSV 格式。"""

    if not path.exists():
        raise FileNotFoundError(f"Missing result CSV: {path}")
    frame = pd.read_csv(path)
    if "Round" in frame.columns:
        round_values = frame["Round"].astype(str).str.strip().str.lower()
        row = frame.loc[round_values.isin({"3", "r3"})]
    elif "Stage" in frame.columns:
        stage_values = frame["Stage"].astype(str).str.strip().str.lower()
        row = frame.loc[stage_values.isin({"after r3", "r3"})]
    else:
        raise KeyError(f"{path} has neither Round nor Stage column: {list(frame.columns)}")
    if row.empty:
        raise ValueError(f"Missing R3 row in {path}")
    return row.iloc[-1]


def _metric(row: pd.Series, column: str) -> float:
    """把 pandas 标量转成普通 float，便于稳定写入 JSON。"""

    return float(row[column])


def collect_variant(root: Path, variant: str, job_id: str) -> dict[str, object]:
    """收集一个 seed31 变体的最终识别指标和 R3 聚类指标。"""

    save_dir = root / f"lora_s28_seed31_{variant}_{job_id}"
    inc = _r3_row(save_dir / "incremental_results.csv")
    clu = _r3_row(save_dir / "clustering_results.csv")
    record: dict[str, object] = {
        "variant": variant,
        "save_dir": str(save_dir),
        "overall": _metric(inc, "Overall Acc"),
        "old": _metric(inc, "Old Acc"),
        "new": _metric(inc, "New Acc"),
        "forgetting": _metric(inc, "Forgetting Rate"),
        "macro_f1": _metric(inc, "Macro F1"),
        "ari": _metric(clu, "ARI"),
        "purity": _metric(clu, "Purity"),
        "hungarian": _metric(clu, "Hungarian Acc"),
    }
    record["delta_overall"] = float(record["overall"]) - BASELINE_SEED31["overall"]
    record["delta_old"] = float(record["old"]) - BASELINE_SEED31["old"]
    record["delta_new"] = float(record["new"]) - BASELINE_SEED31["new"]
    record["delta_forgetting"] = float(record["forgetting"]) - BASELINE_SEED31["forgetting"]
    record["passed_seed31_rescue_gate"] = (
        record["new"] >= BASELINE_SEED31["new"] + 0.05
        and record["overall"] >= BASELINE_SEED31["overall"]
        and record["old"] >= BASELINE_SEED31["old"] - 0.02
        and record["forgetting"] <= BASELINE_SEED31["forgetting"] + 0.02
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage47 LoRa s28 seed31 新类修复报告器")
    parser.add_argument("--root", default="results/stage47")
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "rec065", "rec075", "top080", "rec065_top080"],
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    records = [collect_variant(root, variant, str(args.job_id)) for variant in args.variants]
    best = max(records, key=lambda item: (float(item["new"]), float(item["overall"])))
    verdict = (
        f"{best['variant']} 通过 seed31 修复门槛，可进入三种子确认。"
        if best["passed_seed31_rescue_gate"]
        else "未通过 seed31 修复门槛，暂不扩三种子。"
    )

    lines = [
        f"# Stage47 LoRa s28 Seed31 新类修复报告（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- 当前判定：{verdict}",
        f"- Seed31 对照 R3 Overall/Old/New/Forgetting = "
        f"{BASELINE_SEED31['overall']:.4f}/{BASELINE_SEED31['old']:.4f}/{BASELINE_SEED31['new']:.4f}/{BASELINE_SEED31['forgetting']:.4f}",
        "",
        "## R3 指标",
        "",
        "| Variant | Overall | Old | New | Forgetting | ΔOverall | ΔOld | ΔNew | ΔForgetting | ARI | Hungarian | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in records:
        lines.append(
            "| {variant} | {overall:.4f} | {old:.4f} | {new:.4f} | {forgetting:.4f} | "
            "{delta_overall:+.4f} | {delta_old:+.4f} | {delta_new:+.4f} | {delta_forgetting:+.4f} | "
            "{ari:.4f} | {hungarian:.4f} | {gate} |".format(
                variant=item["variant"],
                overall=item["overall"],
                old=item["old"],
                new=item["new"],
                forgetting=item["forgetting"],
                delta_overall=item["delta_overall"],
                delta_old=item["delta_old"],
                delta_new=item["delta_new"],
                delta_forgetting=item["delta_forgetting"],
                ari=item["ari"],
                hungarian=item["hungarian"],
                gate="PASS" if item["passed_seed31_rescue_gate"] else "FAIL",
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "job_id": str(args.job_id),
                "baseline_seed31": BASELINE_SEED31,
                "records": records,
                "best_variant": best["variant"],
                "verdict": verdict,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
