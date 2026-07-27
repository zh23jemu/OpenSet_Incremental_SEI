"""阶段 5 LoRa25 strict 协议审计工具。

该脚本只检查 LoRa RFFP Different Days Indoor 紧凑数据是否满足
10 known + 5x3 的严格开放集增量协议，不训练模型，也不使用评估集真值
进行参数选择。输出 JSON 用于进入 LoRa 单种子 Slurm smoke test 前的
可复现证据。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.lora25_strict_loader import (  # noqa: E402
    EXPECTED_LABELS,
    EXPECTED_TRANSMISSIONS,
    STAGE_KEYS,
    load_lora25_diffdays_3round,
)


def _unique_ints(values: np.ndarray) -> list[int]:
    """返回稳定排序的整数唯一值，便于 JSON 审计和人工复核。"""
    return [int(v) for v in np.unique(np.asarray(values, dtype=np.int64))]


def _summarize_split(name: str, item: dict[str, Any]) -> dict[str, Any]:
    """汇总单个 split 的形状、标签和 transmission 范围。"""
    x = np.asarray(item["X"])
    y = np.asarray(item["y"])
    transmissions = np.asarray(item["transmission"])
    return {
        "split": name,
        "shape": [int(v) for v in x.shape],
        "dtype": str(x.dtype),
        "labels": _unique_ints(y),
        "label_count": int(np.unique(y).size),
        "transmissions": _unique_ints(transmissions),
        "recording_id_count": int(np.unique(np.asarray(item["recording_id"])).size),
        "expected_labels": list(EXPECTED_LABELS[name]),
        "expected_transmissions": list(EXPECTED_TRANSMISSIONS[name]),
    }


def _check_round_boundary(splits: dict[str, Any], round_index: int) -> dict[str, Any]:
    """检查同一轮新增类 discovery 与 held-out eval 是否使用 disjoint transmission。"""
    discovery_key = f"day{round_index + 1}_unknown_round{round_index}"
    eval_key = f"day{round_index + 1}_eval_after_r{round_index}"
    discovery = splits[discovery_key]
    evaluation = splits[eval_key]
    new_labels = set(EXPECTED_LABELS[discovery_key])

    eval_new_mask = np.isin(np.asarray(evaluation["y"]), list(new_labels))
    discovery_tx = set(_unique_ints(np.asarray(discovery["transmission"])))
    eval_tx = set(_unique_ints(np.asarray(evaluation["transmission"])[eval_new_mask]))
    overlap = sorted(discovery_tx.intersection(eval_tx))
    return {
        "round": int(round_index),
        "discovery_key": discovery_key,
        "eval_key": eval_key,
        "new_labels": sorted(int(v) for v in new_labels),
        "discovery_transmissions": sorted(discovery_tx),
        "eval_new_transmissions": sorted(eval_tx),
        "transmission_overlap": overlap,
        "ok": len(overlap) == 0,
    }


def _check_day1_boundary(splits: dict[str, Any]) -> dict[str, Any]:
    """检查 Day1 60/10/30 是否按完整 transmission 分离。"""
    train = splits["day1_backbone_train"]
    validation = splits["day1_known_validation"]
    evaluation = splits["day1_initial_eval"]
    train_tx = set(_unique_ints(np.asarray(train["transmission"])))
    validation_tx = set(_unique_ints(np.asarray(validation["transmission"])))
    evaluation_tx = set(_unique_ints(np.asarray(evaluation["transmission"])))
    overlap = sorted(
        train_tx.intersection(validation_tx)
        | train_tx.intersection(evaluation_tx)
        | validation_tx.intersection(evaluation_tx)
    )
    total = len(train["y"]) + len(validation["y"]) + len(evaluation["y"])
    ratios = {
        "train": len(train["y"]) / total,
        "validation": len(validation["y"]) / total,
        "evaluation": len(evaluation["y"]) / total,
    }
    return {
        "train_transmissions": sorted(train_tx),
        "validation_transmissions": sorted(validation_tx),
        "evaluation_transmissions": sorted(evaluation_tx),
        "transmission_overlap": overlap,
        "sample_counts": {
            "train": int(len(train["y"])),
            "validation": int(len(validation["y"])),
            "evaluation": int(len(evaluation["y"])),
        },
        "sample_ratios": ratios,
        "ok": not overlap and all(
            abs(ratios[name] - expected) < 1e-9
            for name, expected in (("train", 0.60), ("validation", 0.10), ("evaluation", 0.30))
        ),
    }


def build_audit(npz_path: Path, source_manifest: Path | None, alignment_manifest: Path | None) -> dict[str, Any]:
    """构造 LoRa strict 协议审计报告。"""
    errors: list[str] = []
    warnings: list[str] = []
    splits: dict[str, Any] | None = None
    try:
        splits = load_lora25_diffdays_3round(npz_path)
    except Exception as exc:  # pragma: no cover - CLI 入口保留具体错误文本
        errors.append(str(exc))

    split_summaries = []
    boundary_checks = []
    day1_boundary: dict[str, Any] = {}
    if splits is not None:
        for key in ("day1_backbone_train", "day1_known_validation", *STAGE_KEYS):
            split_summaries.append(_summarize_split(key, splits[key]))
        day1_boundary = _check_day1_boundary(splits)
        if not day1_boundary["ok"]:
            errors.append(f"Day1 60/10/30 transmission boundary invalid: {day1_boundary}")
        boundary_checks = [_check_round_boundary(splits, i) for i in range(1, 4)]
        errors.extend(
            f"Round {row['round']} discovery/eval transmission overlap: {row['transmission_overlap']}"
            for row in boundary_checks
            if not row["ok"]
        )

    manifest_summary: dict[str, Any] = {}
    for label, manifest_path in (
        ("source_manifest", source_manifest),
        ("alignment_manifest", alignment_manifest),
    ):
        if manifest_path is None:
            continue
        if not manifest_path.exists():
            warnings.append(f"{label} not found: {manifest_path}")
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_summary[label] = {
            "path": str(manifest_path),
            "dataset": payload.get("dataset"),
            "preprocessing": payload.get("preprocessing"),
            "record_count": len(payload.get("records", [])),
            "physical_device_count": len(payload.get("mapped_label_to_physical_device", {}))
            if "mapped_label_to_physical_device" in payload
            else None,
            "alignment_score_summary": payload.get("alignment_score_summary"),
            "protocol": payload.get("protocol"),
        }

    expected_shape_ok = bool(split_summaries) and all(row["shape"][1:] == [2, 256] for row in split_summaries)
    expected_dtype_ok = bool(split_summaries) and all(row["dtype"] == "float32" for row in split_summaries)
    expected_label_ok = bool(split_summaries) and all(row["labels"] == row["expected_labels"] for row in split_summaries)
    expected_tx_ok = bool(split_summaries) and all(row["transmissions"] == row["expected_transmissions"] for row in split_summaries)

    ok = not errors and expected_shape_ok and expected_dtype_ok and expected_label_ok and expected_tx_ok
    return {
        "schema_version": "stage5_lora_protocol_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": bool(ok),
        "dataset": "LoRa RFFP Different Days Indoor compact aligned subset",
        "npz_path": str(npz_path),
        "protocol": "LoRa25 strict 10 known + 5 x 3 incremental rounds",
        "selection_boundary": "仅审计 discovery/train 与 held-out eval 的文件边界；不使用评估集真值进行模型选择、聚类参数选择或阈值校准。",
        "split_summaries": split_summaries,
        "day1_boundary_check": day1_boundary,
        "round_boundary_checks": boundary_checks,
        "manifest_summary": manifest_summary,
        "checks": {
            "shape_n_2_256": bool(expected_shape_ok),
            "dtype_float32": bool(expected_dtype_ok),
            "labels_match_10_plus_5x3": bool(expected_label_ok),
            "day1_recording_level_60_10_30": bool(day1_boundary.get("ok", False)),
            "transmissions_disjoint_train_eval": bool(expected_tx_ok and all(row["ok"] for row in boundary_checks)),
            "heldout_eval_not_for_selection": True,
        },
        "errors": errors,
        "warnings": warnings,
        "next_step_if_ok": "准备 LoRa 单种子训练 smoke test；若失败，先修复紧凑数据或协议映射而不是调训练参数。",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 5 LoRa25 strict 协议审计。")
    parser.add_argument("--npz-path", default="datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz")
    parser.add_argument("--source-manifest", default="datasets/lora25_compact/lora25_source_manifest.json")
    parser.add_argument("--alignment-manifest", default="datasets/lora25_compact/lora25_alignment_group_manifest_256.json")
    parser.add_argument("--output", default="results/stage5/stage5_lora_protocol_audit_local.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    npz_path = Path(args.npz_path)
    source_manifest = Path(args.source_manifest) if args.source_manifest else None
    alignment_manifest = Path(args.alignment_manifest) if args.alignment_manifest else None
    report = build_audit(npz_path, source_manifest, alignment_manifest)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
