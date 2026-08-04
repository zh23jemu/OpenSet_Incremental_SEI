"""汇总 Stage45 LoRa 原始 I/Q 长窗/多窗 seed7 验证结果。

Stage45 的目标不是继续在 9.7MB compact 子集上调小机制，而是确认完整
Setup 1 原始 I/Q 重新切窗后，是否能给 LoRa Chirp + GPCC + RADCIL 主线
带来可传递到增量识别的收益。报告器只读取实验输出 CSV，不读取任何 held-out
eval 之外的额外真值，也不参与训练或调参。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASELINE = {
    "overall": 0.2543,
    "old": 0.1929,
    "new": 0.5000,
    "forgetting": 0.2690,
}


def _r3_row(path: Path) -> pd.Series:
    """读取增量结果的最后一轮指标；兼容不同阶段脚本的 R3 标识方式。

    历史工具里有两种 CSV 格式：部分报告表保留数值列 ``Round``，而主实验
    ``incremental_results.csv`` 使用文本列 ``Stage``（例如 ``After R3``）。
    Stage45 只需要最终 R3 指标，因此这里按列名自适应定位，避免因为报告器
    口径不一致把已经完成的训练误判为失败。
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing result CSV: {path}")
    frame = pd.read_csv(path)
    if "Round" in frame.columns:
        row = frame.loc[frame["Round"] == 3]
    elif "Stage" in frame.columns:
        stage_values = frame["Stage"].astype(str).str.strip().str.lower()
        row = frame.loc[stage_values.isin({"after r3", "r3"})]
    else:
        raise KeyError(f"{path} has neither Round nor Stage column: {list(frame.columns)}")
    if row.empty:
        raise ValueError(f"Missing R3 row in {path}")
    return row.iloc[-1]


def _metric(row: pd.Series, name: str) -> float:
    """把 pandas 标量统一转成普通 float，方便 JSON/Markdown 输出。"""

    return float(row[name])


def collect_variant(root: Path, variant: str, job_id: str) -> dict[str, object]:
    """收集单个多窗配置的 symbol-level 和 recording-level R3 指标。"""

    save_dir = root / f"lora_raw_{variant}_seed7_{job_id}"
    symbol = _r3_row(save_dir / "incremental_results.csv")
    record: dict[str, object] = {
        "variant": variant,
        "save_dir": str(save_dir),
        "overall": _metric(symbol, "Overall Acc"),
        "old": _metric(symbol, "Old Acc"),
        "new": _metric(symbol, "New Acc"),
        "forgetting": _metric(symbol, "Forgetting Rate"),
        "macro_f1": _metric(symbol, "Macro F1"),
    }

    recording_csv = save_dir / "recording_level_incremental_results.csv"
    if recording_csv.exists():
        recording = _r3_row(recording_csv)
        record["recording_overall"] = _metric(recording, "Overall Acc")
        record["recording_new"] = _metric(recording, "New Acc")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage45 LoRa 原始 I/Q 多窗报告器")
    parser.add_argument("--root", default="results/stage45")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--variants", nargs="+", default=["s14", "s28"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    records = [collect_variant(root, variant, str(args.job_id)) for variant in args.variants]
    for item in records:
        item["delta_overall"] = float(item["overall"]) - BASELINE["overall"]
        item["delta_old"] = float(item["old"]) - BASELINE["old"]
        item["delta_new"] = float(item["new"]) - BASELINE["new"]
        item["delta_forgetting"] = float(item["forgetting"]) - BASELINE["forgetting"]
        item["passed_seed7_gate"] = (
            item["overall"] > BASELINE["overall"]
            and item["old"] >= BASELINE["old"]
            and item["new"] >= BASELINE["new"] - 0.02
            and item["forgetting"] <= BASELINE["forgetting"] + 0.02
        )

    best = max(records, key=lambda item: float(item["overall"]))
    verdict = (
        f"{best['variant']} 通过 seed7 门槛，可进入三种子。"
        if best["passed_seed7_gate"]
        else "未通过 seed7 门槛，暂不扩三种子。"
    )

    lines = [
        f"# Stage45 LoRa 原始 I/Q 长窗/多窗 Seed7 报告（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- 当前判定：{verdict}",
        f"- 对照：Stage27 Chirp seed7 R3 Overall/Old/New/Forgetting = "
        f"{BASELINE['overall']:.4f}/{BASELINE['old']:.4f}/{BASELINE['new']:.4f}/{BASELINE['forgetting']:.4f}",
        "",
        "## R3 指标",
        "",
        "| Variant | Overall | Old | New | Forgetting | ΔOverall | ΔOld | ΔNew | ΔForgetting | Recording Overall | Recording New | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in records:
        lines.append(
            "| {variant} | {overall:.4f} | {old:.4f} | {new:.4f} | {forgetting:.4f} | "
            "{delta_overall:+.4f} | {delta_old:+.4f} | {delta_new:+.4f} | {delta_forgetting:+.4f} | "
            "{recording_overall} | {recording_new} | {gate} |".format(
                variant=item["variant"],
                overall=item["overall"],
                old=item["old"],
                new=item["new"],
                forgetting=item["forgetting"],
                delta_overall=item["delta_overall"],
                delta_old=item["delta_old"],
                delta_new=item["delta_new"],
                delta_forgetting=item["delta_forgetting"],
                recording_overall=(
                    f"{item['recording_overall']:.4f}"
                    if "recording_overall" in item
                    else "-"
                ),
                recording_new=(
                    f"{item['recording_new']:.4f}"
                    if "recording_new" in item
                    else "-"
                ),
                gate="PASS" if item["passed_seed7_gate"] else "FAIL",
            )
        )

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 输入来自 Stage44 已下载的完整 Setup 1 原始 I/Q，本阶段只重新切窗/对齐，不改变 strict split。",
            "- 若 seed7 不过门槛，不继续三种子，避免把大数据路线变成新的小参数搜索。",
            "",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "job_id": str(args.job_id),
                "baseline": BASELINE,
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
