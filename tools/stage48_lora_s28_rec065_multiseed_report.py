"""汇总 Stage48 LoRa s28 + recording-consensus 0.65 三种子确认结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


STAGE27_BASELINE = {"overall": 0.2540, "old": 0.1952, "new": 0.4889, "forgetting": 0.2794}
STAGE46_S28 = {"overall": 0.2806, "old": 0.2413, "new": 0.4381, "forgetting": 0.1643}


def _r3_row(path: Path) -> pd.Series:
    """读取最终 R3 行，兼容主实验的 ``Stage=After R3`` 和报告表的 ``Round=R3``。"""

    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "Stage" in frame.columns:
        values = frame["Stage"].astype(str).str.strip().str.lower()
        row = frame.loc[values.isin({"after r3", "r3"})]
    elif "Round" in frame.columns:
        values = frame["Round"].astype(str).str.strip().str.lower()
        row = frame.loc[values.isin({"3", "r3"})]
    else:
        raise KeyError(f"{path} has neither Stage nor Round column: {list(frame.columns)}")
    if row.empty:
        raise ValueError(f"Missing R3 row in {path}")
    return row.iloc[-1]


def _metric(row: pd.Series, column: str) -> float:
    """返回普通 float，避免 numpy 标量影响 JSON 序列化。"""

    return float(row[column])


def collect_seed(root: Path, seed: int, job_id: str) -> dict[str, object]:
    """收集单个 seed 的最终 R3 symbol-level 和 recording-level 指标。"""

    save_dir = root / f"lora_s28_rec065_seed{seed}_{job_id}"
    inc = _r3_row(save_dir / "incremental_results.csv")
    item: dict[str, object] = {
        "seed": int(seed),
        "save_dir": str(save_dir),
        "overall": _metric(inc, "Overall Acc"),
        "old": _metric(inc, "Old Acc"),
        "new": _metric(inc, "New Acc"),
        "forgetting": _metric(inc, "Forgetting Rate"),
        "macro_f1": _metric(inc, "Macro F1"),
    }
    recording_csv = save_dir / "recording_level_incremental_results.csv"
    if recording_csv.exists():
        rec = _r3_row(recording_csv)
        item["recording_overall"] = _metric(rec, "Overall Acc")
        item["recording_new"] = _metric(rec, "New Acc")
    return item


def mean_std(records: list[dict[str, object]], key: str) -> tuple[float, float]:
    """计算均值和样本标准差；缺失 recording 指标时自动跳过。"""

    values = [float(item[key]) for item in records if key in item]
    series = pd.Series(values, dtype=float)
    return float(series.mean()), float(series.std(ddof=1) if len(series) > 1 else 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage48 LoRa s28 rec065 三种子报告器")
    parser.add_argument("--root", default="results/stage48")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 31])
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    records = [collect_seed(root, seed, str(args.job_id)) for seed in args.seeds]
    summary = {
        key: mean_std(records, key)
        for key in ("overall", "old", "new", "forgetting", "macro_f1", "recording_overall", "recording_new")
        if any(key in item for item in records)
    }
    delta_stage27 = {key: summary[key][0] - STAGE27_BASELINE[key] for key in STAGE27_BASELINE}
    delta_stage46 = {key: summary[key][0] - STAGE46_S28[key] for key in STAGE46_S28}
    passed = (
        delta_stage27["overall"] > 0
        and delta_stage27["old"] > 0
        and delta_stage27["new"] >= -0.02
        and delta_stage27["forgetting"] <= 0.02
        and delta_stage46["new"] > 0
    )

    def fmt(key: str) -> str:
        mean, std = summary[key]
        return f"{mean:.4f}±{std:.4f}"

    lines = [
        f"# Stage48 LoRa s28 rec065 三种子报告（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'通过三种子门槛，可作为 LoRa 新正式候选。' if passed else '未通过三种子门槛，暂不替换正式候选。'}",
        f"- Stage48 R3 Overall/Old/New/Forgetting = {fmt('overall')}/{fmt('old')}/{fmt('new')}/{fmt('forgetting')}",
        f"- vs Stage27 Chirp：Overall {delta_stage27['overall']:+.4f}，Old {delta_stage27['old']:+.4f}，New {delta_stage27['new']:+.4f}，Forgetting {delta_stage27['forgetting']:+.4f}",
        f"- vs Stage46 s28：Overall {delta_stage46['overall']:+.4f}，Old {delta_stage46['old']:+.4f}，New {delta_stage46['new']:+.4f}，Forgetting {delta_stage46['forgetting']:+.4f}",
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
                recording_overall=f"{item['recording_overall']:.4f}" if "recording_overall" in item else "-",
                recording_new=f"{item['recording_new']:.4f}" if "recording_new" in item else "-",
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "job_id": str(args.job_id),
                "seeds": [int(seed) for seed in args.seeds],
                "stage27_baseline": STAGE27_BASELINE,
                "stage46_s28": STAGE46_S28,
                "records": records,
                "summary": {key: {"mean": value[0], "std": value[1]} for key, value in summary.items()},
                "delta_vs_stage27": delta_stage27,
                "delta_vs_stage46": delta_stage46,
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
