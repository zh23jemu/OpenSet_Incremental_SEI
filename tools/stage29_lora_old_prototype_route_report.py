"""汇总 Stage 29 LoRa 旧类原型路由 seed7 结果。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _rows(path: Path) -> list[dict[str, str]]:
    """读取 CSV 结果文件。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"结果为空：{path}")
    return rows


def _r3(path: Path) -> dict[str, str]:
    """取 After R3 行。"""

    rows = _rows(path)
    for row in rows:
        if str(row.get("Stage", "")).strip().lower() == "after r3":
            return row
    return rows[-1]


def _float(row: dict[str, str], key: str) -> float:
    """解析指标。"""

    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 Stage 29 LoRa 旧类原型路由 seed7。")
    parser.add_argument("--root", default="results/stage29")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    save_dir = Path(args.root) / f"lora_old_prototype_route_seed7_{args.job_id}"
    r3 = _r3(save_dir / "incremental_results.csv")
    record_r3 = _r3(save_dir / "recording_level_incremental_results.csv")
    calibration_path = save_dir / "old_prototype_route_iq7_calibration.csv"
    calibration_rows = _rows(calibration_path) if calibration_path.exists() else []

    result = {
        "overall": _float(r3, "Overall Acc"),
        "old": _float(r3, "Old Acc"),
        "new": _float(r3, "New Acc"),
        "forgetting": _float(r3, "Forgetting Rate"),
        "macro_f1": _float(r3, "Macro F1"),
        "recording_overall": _float(record_r3, "Overall Acc"),
        "recording_new": _float(record_r3, "New Acc"),
    }
    references = {
        "stage27_chirp_radcil_seed7": {
            "overall": 0.2581,
            "old": 0.1988,
            "new": 0.4952,
            "forgetting": 0.2690,
        },
        "stage28_chirp_doi_style_seed7": {
            "overall": 0.2943,
            "old": 0.3131,
            "new": 0.2190,
            "forgetting": 0.1357,
        },
    }
    delta_vs_radcil = {
        key: result[key] - references["stage27_chirp_radcil_seed7"][key]
        for key in ("overall", "old", "new", "forgetting")
    }
    passed = (
        result["overall"] >= references["stage27_chirp_radcil_seed7"]["overall"] + 0.01
        and result["old"] >= references["stage27_chirp_radcil_seed7"]["old"] + 0.03
        and result["new"] >= references["stage27_chirp_radcil_seed7"]["new"] - 0.08
    )
    verdict = "通过 seed7 门槛，可扩三种子。" if passed else "未通过 seed7 门槛，先归档诊断结果。"

    lines = [
        "# Stage 29 LoRa 旧类原型路由 seed7 报告",
        "",
        f"- Slurm Job：`{args.job_id}`",
        "- 方法：Chirp + GPCC + cross-day + LoRa SSL + joint discovery-CIL，评估时对高旧类概率样本使用旧类 replay prototype 路由。",
        "- 校准：只使用 Day1/IQ_7 旧类验证集选择 old-mass 阈值；held-out IQ_8-10 仅用于最终评估。",
        "",
        "| Method | R3 Overall | R3 Old | R3 New | Forgetting | Macro F1 | Rec Overall | Rec New |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| Old-prototype route | {_fmt(result['overall'])} | {_fmt(result['old'])} | "
            f"{_fmt(result['new'])} | {_fmt(result['forgetting'])} | {_fmt(result['macro_f1'])} | "
            f"{_fmt(result['recording_overall'])} | {_fmt(result['recording_new'])} |"
        ),
        (
            "| Stage27 RADCIL ref | "
            f"{_fmt(references['stage27_chirp_radcil_seed7']['overall'])} | "
            f"{_fmt(references['stage27_chirp_radcil_seed7']['old'])} | "
            f"{_fmt(references['stage27_chirp_radcil_seed7']['new'])} | "
            f"{_fmt(references['stage27_chirp_radcil_seed7']['forgetting'])} | nan | nan | nan |"
        ),
        (
            "| Stage28 DOI-style ref | "
            f"{_fmt(references['stage28_chirp_doi_style_seed7']['overall'])} | "
            f"{_fmt(references['stage28_chirp_doi_style_seed7']['old'])} | "
            f"{_fmt(references['stage28_chirp_doi_style_seed7']['new'])} | "
            f"{_fmt(references['stage28_chirp_doi_style_seed7']['forgetting'])} | nan | nan | nan |"
        ),
        "",
        "## 相对 Stage27 RADCIL",
        "",
        f"- Overall `{_fmt(delta_vs_radcil['overall'])}`、Old `{_fmt(delta_vs_radcil['old'])}`、New `{_fmt(delta_vs_radcil['new'])}`、Forgetting `{_fmt(delta_vs_radcil['forgetting'])}`。",
        f"- 当前判定：{verdict}",
        "",
    ]
    if calibration_rows:
        lines.extend(["## IQ_7 路由校准", ""])
        for row in calibration_rows:
            lines.append(
                f"- {row.get('Stage')}: threshold `{row.get('Old Mass Threshold')}` -> validation acc `{row.get('Validation Accuracy')}`"
            )
        lines.append("")

    summary = {
        "job_id": str(args.job_id),
        "save_dir": str(save_dir),
        "result": result,
        "references": references,
        "delta_vs_stage27_radcil": delta_vs_radcil,
        "passed": passed,
        "verdict": verdict,
        "calibration_rows": calibration_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
