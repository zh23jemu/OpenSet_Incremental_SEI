"""汇总 Stage 38 LoRa 物理域自监督预训练三种子结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


# Stage 27 Chirp backbone 是当前 LoRa 正式候选的三种子口径；Stage 38 必须
# 在这个候选上继续提高，同时不能靠牺牲 New Acc 换取表面 Overall。
STAGE27_CHIRP_BASELINE = {
    "overall": 0.2540,
    "old": 0.1952,
    "new": 0.4889,
    "forgetting": 0.2794,
    "recording_overall": 0.2933,
    "recording_new": 0.6444,
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    """读取 CSV 指标文件；缺文件直接报错，避免静默生成假报告。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"结果为空：{path}")
    return rows


def _r3(path: Path) -> dict[str, str]:
    """优先读取 AFTER R3 行，兼容只有最终行的旧格式结果。"""

    rows = _read_rows(path)
    for row in rows:
        if str(row.get("Stage", "")).strip().lower() == "after r3":
            return row
    return rows[-1]


def _float(row: dict[str, str], key: str) -> float:
    """解析指标列；缺失时返回 NaN 并在报告中显式展示。"""

    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _mean_std(values: list[float]) -> tuple[float, float]:
    """计算总体均值和标准差，忽略 NaN 以便定位单项缺失。"""

    valid = [value for value in values if not math.isnan(value)]
    if not valid:
        return float("nan"), float("nan")
    mean = sum(valid) / len(valid)
    std = math.sqrt(sum((value - mean) ** 2 for value in valid) / len(valid))
    return mean, std


def _fmt(value: float) -> str:
    """统一指标格式。"""

    return "nan" if math.isnan(value) else f"{value:.4f}"


def _checkpoint_best(root: Path, seed: int, job_id: str) -> dict[str, float | int]:
    """读取对应 seed 的 closed-set checkpoint 选择结果。"""

    summary_path = root / f"stage38_lora_physical_pretrain_checkpoint_summary_seed{seed}_{job_id}.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best = summary["best"]
    return {
        "epoch": int(best["epoch"]),
        "validation_accuracy": float(best["validation_accuracy"]),
        "validation_loss": float(best["validation_loss"]),
    }


def main() -> int:
    """命令行入口，生成 Markdown 报告和 JSON summary。"""

    parser = argparse.ArgumentParser(description="汇总 Stage 38 LoRa physical pretrain 三种子结果")
    parser.add_argument("--root", default="results/stage38")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--seeds", default="7,13,31")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    records: list[dict[str, object]] = []
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    for seed in seeds:
        save_dir = root / f"lora_physical_pretrain_seed{seed}_{args.job_id}"
        symbol = _r3(save_dir / "incremental_results.csv")
        recording = _r3(save_dir / "recording_level_incremental_results.csv")
        best = _checkpoint_best(root, seed, str(args.job_id))
        records.append(
            {
                "seed": seed,
                "checkpoint_epoch": best["epoch"],
                "checkpoint_val_acc": best["validation_accuracy"],
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
        "overall",
        "old",
        "new",
        "forgetting",
        "macro_f1",
        "recording_overall",
        "recording_new",
    )
    mean_std: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        mean, std = _mean_std([float(record[key]) for record in records])
        mean_std[key] = {"mean": mean, "std": std}

    # 通过标准保持克制：需要 Overall/Old 超过当前正式候选，且 New 不能明显塌缩。
    passed = (
        mean_std["overall"]["mean"] >= STAGE27_CHIRP_BASELINE["overall"] + 0.005
        and mean_std["old"]["mean"] >= STAGE27_CHIRP_BASELINE["old"]
        and mean_std["new"]["mean"] >= STAGE27_CHIRP_BASELINE["new"] - 0.02
    )
    verdict = (
        "通过三种子门槛，可作为 LoRa 当前更强候选。"
        if passed
        else "未通过三种子门槛，仅保留为 seed7 正信号或负消融。"
    )

    lines = [
        "# Stage 38 LoRa 物理域自监督预训练三种子报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 方法：每个 seed 重新训练 LoRa Chirp 初始 checkpoint，目标为 CE + SupCon + 物理描述符回归。",
        "- 固定项：strict split、GPCC recording-consensus 0.65、cross-day、LoRa SSL、top0.80 注册和 RADCIL 后端。",
        "- 边界：checkpoint 训练只用 Day1 IQ_1-6 与 IQ_7，不读取 Day1 IQ_8-10 或后续 held-out eval。",
        "- 对照：Stage 27 当前正式 LoRa 候选三种子 Overall/Old/New/Forgetting = `0.2540/0.1952/0.4889/0.2794`。",
        "",
        "| Seed | Ckpt Epoch | IQ_7 Val | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 | Recording Overall | Recording New |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['seed']} | {record['checkpoint_epoch']} | "
            f"{_fmt(float(record['checkpoint_val_acc']))} | {_fmt(float(record['overall']))} | "
            f"{_fmt(float(record['old']))} | {_fmt(float(record['new']))} | "
            f"{_fmt(float(record['forgetting']))} | {_fmt(float(record['macro_f1']))} | "
            f"{_fmt(float(record['recording_overall']))} | {_fmt(float(record['recording_new']))} |"
        )

    labels = {
        "overall": "R3 Overall",
        "old": "R3 Old",
        "new": "R3 New",
        "forgetting": "Forgetting",
        "macro_f1": "Macro F1",
        "recording_overall": "Recording Overall",
        "recording_new": "Recording New",
    }
    lines.extend(["", "## 三种子均值 ± 标准差", ""])
    for key, label in labels.items():
        lines.append(f"- {label}: {_fmt(mean_std[key]['mean'])} ± {_fmt(mean_std[key]['std'])}")
    lines.extend(["", f"- 当前判定：{verdict}", ""])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "schema_version": "stage38_lora_physical_pretrain_multiseed_summary_v1",
        "job_id": str(args.job_id),
        "seeds": seeds,
        "records": records,
        "mean_std": mean_std,
        "baseline": STAGE27_CHIRP_BASELINE,
        "passed": bool(passed),
        "verdict": verdict,
    }
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Stage 38 三种子报告：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
