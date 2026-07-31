"""汇总 LoRa Stage 24 recording-level GPCC 三种子结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    """读取 CSV 结果；缺文件或空文件都视为实验未完成。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"结果为空：{path}")
    return rows


def _last_r3(path: Path) -> dict[str, str]:
    """读取增量结果中的 After R3 行。"""

    rows = _read_rows(path)
    for row in rows:
        if str(row.get("Stage", "")).strip().upper() == "AFTER R3":
            return row
    return rows[-1]


def _float(row: dict[str, str], key: str) -> float:
    """把指标列转成 float，缺失时返回 NaN 便于报告暴露问题。"""

    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _mean_std(values: list[float]) -> tuple[float, float]:
    """计算总体均值和标准差。"""

    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return float("nan"), float("nan")
    mean = sum(clean) / len(clean)
    return mean, math.sqrt(sum((value - mean) ** 2 for value in clean) / len(clean))


def _fmt(value: float) -> str:
    """统一指标展示格式。"""

    return "nan" if math.isnan(value) else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 LoRa Stage 24 recording-level GPCC 三种子结果。")
    parser.add_argument("--root", default="results/stage24")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--seeds", default="7,13,31")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    records = []
    for seed_text in args.seeds.split(","):
        seed = int(seed_text.strip())
        save_dir = Path(args.root) / f"lora_recording_gpcc_seed{seed}_{args.job_id}"
        row = _last_r3(save_dir / "incremental_results.csv")
        records.append(
            {
                "seed": seed,
                "overall": _float(row, "Overall Acc"),
                "old": _float(row, "Old Acc"),
                "new": _float(row, "New Acc"),
                "forgetting": _float(row, "Forgetting Rate"),
                "macro_f1": _float(row, "Macro F1"),
            }
        )

    lines = [
        "# Stage 24 LoRa Recording-GPCC 三种子报告",
        "",
        "组合：recording-level GPCC + cross_day 表征适配 + 第一次 CIL + Student 重新发现/无标签 Hungarian 对齐 + 第二次 CIL。",
        "边界：recording_id 是可观测 transmission 元数据，不是未知设备标签；IQ_8-10 held-out evaluation 不参与聚类或训练。",
        "",
        "| Seed | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['seed']} | {_fmt(record['overall'])} | {_fmt(record['old'])} | "
            f"{_fmt(record['new'])} | {_fmt(record['forgetting'])} | {_fmt(record['macro_f1'])} |"
        )

    mean_std = {}
    for key, label in (
        ("overall", "R3 Overall"),
        ("old", "R3 Old"),
        ("new", "R3 New"),
        ("forgetting", "Forgetting"),
        ("macro_f1", "Macro F1"),
    ):
        mean, std = _mean_std([float(record[key]) for record in records])
        mean_std[key] = {"mean": mean, "std": std}
        lines.append(f"- {label}: {_fmt(mean)} ± {_fmt(std)}")

    baseline = {"overall": 0.1667, "old": 0.0917, "new": 0.4667, "forgetting": 0.4857}
    passed = (
        mean_std["overall"]["mean"] > baseline["overall"]
        and mean_std["old"]["mean"] >= baseline["old"] - 0.01
        and mean_std["new"]["mean"] >= baseline["new"] - 0.03
        and mean_std["forgetting"]["mean"] < baseline["forgetting"]
    )
    verdict = (
        "通过三种子候选门槛，可作为 LoRa 当前最强结构候选。"
        if passed
        else "未通过三种子候选门槛，Recording-GPCC 归档为结构性负消融。"
    )
    lines.extend(
        [
            "",
            "对照基线：R3 Overall/Old/New/Forgetting = `0.1667/0.0917/0.4667/0.4857`。",
            f"当前判定：{verdict}",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "job_id": str(args.job_id),
        "records": records,
        "mean_std": mean_std,
        "baseline": baseline,
        "passed": passed,
        "verdict": verdict,
    }
    Path(args.summary_json).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
