"""汇总 Stage46 LoRa 原始 I/Q s28 三种子验证结果。

Stage45 seed7 显示 s28 多符号窗口相对 Stage27 Chirp 候选有明确收益。
本报告器用于确认该收益是否能跨 seed 7/13/31 稳定存在。它只读取每个 seed
已经生成的 CSV 指标，不参与训练、不读取额外真值，也不改变 strict split。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASELINE_MEAN = {
    "overall": 0.2540,
    "old": 0.1952,
    "new": 0.4889,
    "forgetting": 0.2794,
}


def _r3_row(path: Path) -> pd.Series:
    """读取最终 R3 行，兼容 ``Round`` 和 ``Stage=After R3`` 两种表头。"""

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
    """把 pandas 标量转成 JSON 友好的普通 float。"""

    return float(row[name])


def collect_seed(root: Path, seed: int, job_id: str) -> dict[str, object]:
    """收集单个 seed 的 symbol-level 和 recording-level R3 指标。"""

    save_dir = root / f"lora_raw_s28_seed{seed}_{job_id}"
    symbol = _r3_row(save_dir / "incremental_results.csv")
    item: dict[str, object] = {
        "seed": int(seed),
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
        item["recording_overall"] = _metric(recording, "Overall Acc")
        item["recording_new"] = _metric(recording, "New Acc")
    return item


def _mean_std(records: list[dict[str, object]], key: str) -> tuple[float, float]:
    """计算均值和样本标准差，缺失 recording 指标时自动跳过。"""

    values = [float(item[key]) for item in records if key in item]
    series = pd.Series(values, dtype=float)
    return float(series.mean()), float(series.std(ddof=1) if len(series) > 1 else 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage46 LoRa raw s28 三种子报告器")
    parser.add_argument("--root", default="results/stage46")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 31])
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    records = [collect_seed(root, seed, str(args.job_id)) for seed in args.seeds]
    summary_metrics = {
        key: _mean_std(records, key)
        for key in ("overall", "old", "new", "forgetting", "macro_f1", "recording_overall", "recording_new")
        if any(key in item for item in records)
    }
    deltas = {
        key: summary_metrics[key][0] - BASELINE_MEAN[key]
        for key in ("overall", "old", "new", "forgetting")
    }
    passed = (
        deltas["overall"] > 0.0
        and deltas["old"] >= 0.0
        and deltas["new"] >= -0.02
        and deltas["forgetting"] <= 0.02
    )

    def fmt_metric(key: str) -> str:
        mean, std = summary_metrics[key]
        return f"{mean:.4f}±{std:.4f}"

    lines = [
        f"# Stage46 LoRa 原始 I/Q s28 三种子报告（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'通过三种子门槛，可作为新的 LoRa 正式候选。' if passed else '未通过三种子门槛，暂不替换 Stage27 LoRa 候选。'}",
        f"- Stage27 Chirp 三种子对照 R3 Overall/Old/New/Forgetting = "
        f"{BASELINE_MEAN['overall']:.4f}/{BASELINE_MEAN['old']:.4f}/{BASELINE_MEAN['new']:.4f}/{BASELINE_MEAN['forgetting']:.4f}",
        f"- Stage46 s28 三种子 R3 Overall/Old/New/Forgetting = "
        f"{fmt_metric('overall')}/{fmt_metric('old')}/{fmt_metric('new')}/{fmt_metric('forgetting')}",
        "",
        "## Seed 明细",
        "",
        "| Seed | Overall | Old | New | Forgetting | Macro F1 | Recording Overall | Recording New |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in records:
        lines.append(
            "| {seed} | {overall:.4f} | {old:.4f} | {new:.4f} | {forgetting:.4f} | {macro_f1:.4f} | {recording_overall} | {recording_new} |".format(
                seed=item["seed"],
                overall=item["overall"],
                old=item["old"],
                new=item["new"],
                forgetting=item["forgetting"],
                macro_f1=item["macro_f1"],
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
            )
        )

    lines.extend(
        [
            "",
            "## Delta vs Stage27",
            "",
            f"- Overall: {deltas['overall']:+.4f}",
            f"- Old: {deltas['old']:+.4f}",
            f"- New: {deltas['new']:+.4f}",
            f"- Forgetting: {deltas['forgetting']:+.4f}",
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
                "seeds": [int(seed) for seed in args.seeds],
                "baseline_mean": BASELINE_MEAN,
                "records": records,
                "summary_metrics": {
                    key: {"mean": value[0], "std": value[1]}
                    for key, value in summary_metrics.items()
                },
                "delta_vs_stage27": deltas,
                "passed_multiseed_gate": bool(passed),
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
