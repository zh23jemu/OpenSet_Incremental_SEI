"""汇总 Stage 27 LoRa Chirp backbone 三种子结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    """读取单个结果 CSV，并在结果缺失时明确报错。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"结果为空：{path}")
    return rows


def _r3(path: Path) -> dict[str, str]:
    """优先取 After R3 行，兼容旧结果中没有该行的情况。"""

    rows = _read_rows(path)
    for row in rows:
        if str(row.get("Stage", "")).strip().lower() == "after r3":
            return row
    return rows[-1]


def _float(row: dict[str, str], key: str) -> float:
    """安全解析指标列，缺失指标时返回 NaN 供报告标记。"""

    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _mean_std(values: list[float]) -> tuple[float, float]:
    """计算总体均值和标准差，避免单个缺失值中断整份报告。"""

    valid = [value for value in values if not math.isnan(value)]
    if not valid:
        return float("nan"), float("nan")
    mean = sum(valid) / len(valid)
    std = math.sqrt(sum((value - mean) ** 2 for value in valid) / len(valid))
    return mean, std


def _fmt(value: float) -> str:
    """统一报告中的浮点数格式。"""

    return "nan" if math.isnan(value) else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 Stage 27 LoRa Chirp backbone 三种子结果。")
    parser.add_argument("--root", default="results/stage27")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--seeds", default="7,13,31")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    for seed_text in args.seeds.split(","):
        seed = int(seed_text.strip())
        save_dir = Path(args.root) / f"lora_chirp_seed{seed}_{args.job_id}"
        symbol = _r3(save_dir / "incremental_results.csv")
        recording = _r3(save_dir / "recording_level_incremental_results.csv")
        records.append(
            {
                "seed": seed,
                "overall": _float(symbol, "Overall Acc"),
                "old": _float(symbol, "Old Acc"),
                "new": _float(symbol, "New Acc"),
                "forgetting": _float(symbol, "Forgetting Rate"),
                "macro_f1": _float(symbol, "Macro F1"),
                "recording_overall": _float(recording, "Overall Acc"),
                "recording_new": _float(recording, "New Acc"),
            }
        )

    metric_keys = (
        "overall", "old", "new", "forgetting", "macro_f1", "recording_overall", "recording_new"
    )
    mean_std: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        mean, std = _mean_std([float(record[key]) for record in records])
        mean_std[key] = {"mean": mean, "std": std}

    # Stage 27 seed7 的同协议 ResNet1D 对照只作为结构收益参考，不混入三种子均值。
    baseline = {
        "overall": 0.1780952381,
        "old": 0.1071428571,
        "new": 0.4619047619,
        "forgetting": 0.3547619048,
    }
    passed = (
        mean_std["overall"]["mean"] >= baseline["overall"] + 0.02
        and mean_std["old"]["mean"] >= baseline["old"] - 0.01
        and mean_std["new"]["mean"] >= baseline["new"] - 0.03
        and mean_std["forgetting"]["mean"] <= baseline["forgetting"] + 0.05
    )
    verdict = (
        "通过三种子结构门槛，可作为 LoRa 当前正式候选。"
        if passed
        else "未通过三种子结构门槛，Chirp seed7 收益不稳定。"
    )

    lines = [
        "# Stage 27 LoRa Chirp Backbone 三种子报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 方法：每个 seed 重新训练 LoRa-specific Chirp backbone，再接同一 strict split、GPCC、cross-day、LoRa SSL 和联合 discovery-CIL。",
        "- 边界：不读取 held-out eval 真值进行训练、特征选择或阈值选择。",
        "- 参考：Stage 27 seed7 同协议 ResNet1D 对照为 Overall/Old/New/Forgetting = `0.1781/0.1071/0.4619/0.3548`。",
        "",
        "| Seed | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 | Recording Overall | Recording New |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['seed']} | {_fmt(float(record['overall']))} | {_fmt(float(record['old']))} | "
            f"{_fmt(float(record['new']))} | {_fmt(float(record['forgetting']))} | "
            f"{_fmt(float(record['macro_f1']))} | {_fmt(float(record['recording_overall']))} | "
            f"{_fmt(float(record['recording_new']))} |"
        )
    lines.extend(["", "## 三种子均值 ± 标准差", ""])
    labels = {
        "overall": "R3 Overall",
        "old": "R3 Old",
        "new": "R3 New",
        "forgetting": "Forgetting",
        "macro_f1": "Macro F1",
        "recording_overall": "Recording Overall",
        "recording_new": "Recording New",
    }
    for key, label in labels.items():
        lines.append(f"- {label}: {_fmt(mean_std[key]['mean'])} ± {_fmt(mean_std[key]['std'])}")
    lines.extend(["", f"- 当前判定：{verdict}", ""])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "job_id": str(args.job_id),
        "seeds": [int(seed.strip()) for seed in args.seeds.split(",")],
        "records": records,
        "mean_std": mean_std,
        "baseline_seed7_resnet1d": baseline,
        "passed": passed,
        "verdict": verdict,
    }
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
