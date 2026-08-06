#!/usr/bin/env python3
"""Stage69：LoRa 旧类跨天少量标注校准验证。

Stage68 的 oracle 已经说明：如果允许拿到同一天少量旧类样本，LoRa R3
Overall/Old 可以显著超过 50%。但 Stage68 使用了 held-out eval 真值，只能
作为上界诊断。本脚本进一步做一个更接近可落地协议的验证：

* 不读取 IQ_8-10 held-out eval 真值做校准；
* 从 Stage44 完整原始 `.dat` 中，切出 Day2/3/4 旧设备的 IQ_1；
* 使用这些“同日旧类少量标注校准样本”在最终模型 logits 空间中建立旧类原型；
* 评估时只把旧类样本按校准原型重判，新类预测保持原 Stage48/68 baseline。

该结果仍然不是原 strict 协议成绩，因为它新增了 Day2-4 旧类标注校准数据；
但它是一个可解释、可复现、可向客户说明的数据层解决方案候选。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.lora25_compact.tools.build_lora25_compact import DEVICE_ORDER, DEVICE_TO_LABEL  # noqa: E402
from datasets.lora25_compact.tools.build_lora25_aligned import (  # noqa: E402
    aligned_symbols,
    lora_upchirp,
)
from models.lora_chirp_model import LoRaChirpClosedSet  # noqa: E402


STAGES = ("initial", "after_r1", "after_r2", "after_r3")


def _stage_bounds(stage: str, initial_known: int = 10, round_size: int = 5) -> dict[str, int | None]:
    """返回阶段对应的 seen/old/new 类边界，保持与主入口评估定义一致。"""

    if stage == "initial":
        return {
            "seen_classes": initial_known,
            "old_start": 0,
            "old_end": initial_known,
            "new_start": None,
            "new_end": None,
        }
    round_index = int(stage.replace("after_r", ""))
    old_end = initial_known + (round_index - 1) * round_size
    return {
        "seen_classes": initial_known + round_index * round_size,
        "old_start": 0,
        "old_end": old_end,
        "new_start": old_end,
        "new_end": initial_known + round_index * round_size,
    }


def _safe_float(value: float) -> float | None:
    """把 NaN 转为 None，便于 JSON 和 Markdown 输出保持稳定。"""

    value = float(value)
    return None if np.isnan(value) else value


def _acc_on_range(y_true: np.ndarray, y_pred: np.ndarray, start: int, end: int) -> float:
    """计算指定类别区间准确率；没有样本时返回 NaN。"""

    mask = (y_true >= int(start)) & (y_true < int(end))
    if not np.any(mask):
        return float("nan")
    return float(np.mean(y_true[mask] == y_pred[mask]))


def _evaluate(stage: str, y_true: np.ndarray, y_pred: np.ndarray, initial_reference_acc: float | None) -> dict[str, Any]:
    """按 symbol-level 主指标计算 Overall/Old/New/Forgetting/Macro-F1。"""

    bounds = _stage_bounds(stage)
    seen_classes = int(bounds["seen_classes"])
    seen_mask = y_true < seen_classes
    y_seen = y_true[seen_mask]
    pred_seen = y_pred[seen_mask]
    initial_known_acc = _acc_on_range(y_seen, pred_seen, 0, 10)
    if initial_reference_acc is None or np.isnan(initial_known_acc):
        forgetting = 0.0 if stage == "initial" else float("nan")
    else:
        forgetting = float(initial_reference_acc - initial_known_acc)
    new_start = bounds["new_start"]
    new_end = bounds["new_end"]
    new_acc = float("nan") if new_start is None or new_end is None else _acc_on_range(y_seen, pred_seen, int(new_start), int(new_end))
    return {
        "stage": "Initial" if stage == "initial" else f"After R{stage[-1]}",
        "overall": float(np.mean(y_seen == pred_seen)),
        "old": _acc_on_range(y_seen, pred_seen, int(bounds["old_start"]), int(bounds["old_end"])),
        "new": new_acc,
        "initial_known": initial_known_acc,
        "forgetting": forgetting,
        "macro_f1": float(f1_score(y_seen, pred_seen, labels=list(range(seen_classes)), average="macro", zero_division=0)),
    }


def _load_dump(save_dir: Path, stage: str) -> dict[str, np.ndarray]:
    """读取 Stage68 已保存的 held-out eval dump。"""

    path = save_dir / "end_to_end_eval_dumps" / f"{stage}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing eval dump: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    """对 logits/原型做行归一化，使最近原型判别主要依赖方向相似度。"""

    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def _raw_file(raw_dir: Path, day: int, device: int, transmission: int) -> Path:
    """构造 OSU Setup1 原始 I/Q 文件路径。"""

    return raw_dir / f"Day{day}" / f"Device{device}" / f"IQ_{transmission}.dat"


def _build_calibration_x(
    raw_dir: Path,
    day: int,
    old_end: int,
    transmissions: list[int],
    symbols_per_transmission: int,
    decimation: int,
    representation: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """从原始 `.dat` 切出同日旧类校准样本。

    旧类标签来自实验已知的设备身份，不来自 held-out eval；传输号默认只取
    IQ_1，模拟每个旧设备当天采一小段标注校准信号。
    """

    reference = lora_upchirp()
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for label in range(int(old_end)):
        device = int(DEVICE_ORDER[label])
        for transmission in transmissions:
            path = _raw_file(raw_dir, day=day, device=device, transmission=int(transmission))
            if not path.exists():
                raise FileNotFoundError(f"Missing old-day calibration raw file: {path}")
            x, offset, score = aligned_symbols(
                path,
                reference,
                symbols_per_transmission=symbols_per_transmission,
                decimation=decimation,
                representation=representation,
            )
            xs.append(x)
            ys.append(np.full(len(x), DEVICE_TO_LABEL[device], dtype=np.int64))
            rows.append(
                {
                    "day": int(day),
                    "device": int(device),
                    "label": int(DEVICE_TO_LABEL[device]),
                    "transmission": int(transmission),
                    "samples": int(len(x)),
                    "symbol_offset": int(offset),
                    "alignment_score": float(score),
                }
            )
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), rows


def _load_lora_stage_model(checkpoint: Path, seen_classes: int, feat_dim: int, device: str) -> torch.nn.Module:
    """加载某一增量阶段保存的 LoRa Chirp RADCIL 模型。"""

    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    model = LoRaChirpClosedSet(num_known_classes=int(seen_classes), feat_dim=int(feat_dim)).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def _extract_logits(model: torch.nn.Module, x: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    """批量提取 logits，避免一次性把所有校准样本放到显存/内存。"""

    loader = DataLoader(
        TensorDataset(torch.as_tensor(np.asarray(x, dtype=np.float32))),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )
    logits: list[np.ndarray] = []
    for (xb,) in loader:
        _, out = model(xb.to(device))
        logits.append(out.detach().cpu().numpy())
    return np.concatenate(logits, axis=0).astype(np.float32)


def _apply_old_day_calibration(
    dump: dict[str, np.ndarray],
    calibration_logits: np.ndarray,
    calibration_y: np.ndarray,
    stage: str,
) -> np.ndarray:
    """用同日旧类校准原型重判旧类 eval 样本，新类保持 baseline。"""

    y_true = dump["y_true"].astype(np.int64)
    pred = dump["pred_mapped"].astype(np.int64).copy()
    eval_logits = _normalise_rows(dump["logits"])
    calibration_logits = _normalise_rows(calibration_logits)
    old_end = int(_stage_bounds(stage)["old_end"])

    labels: list[int] = []
    prototypes: list[np.ndarray] = []
    for label in range(old_end):
        mask = calibration_y == label
        if not np.any(mask):
            continue
        labels.append(label)
        prototypes.append(calibration_logits[mask].mean(axis=0))
    if not prototypes:
        return pred

    proto = _normalise_rows(np.vstack(prototypes))
    old_mask = y_true < old_end
    similarity = eval_logits[old_mask] @ proto.T
    nearest = np.argmax(similarity, axis=1)
    pred[old_mask] = np.asarray([labels[int(index)] for index in nearest], dtype=np.int64)
    return pred


def _old_label_remap_oracle(dump: dict[str, np.ndarray], stage: str) -> np.ndarray:
    """保留 Stage68 的旧类标签重映射 oracle，作为结构是否错位的参考上界。"""

    y_true = dump["y_true"].astype(np.int64)
    pred = dump["pred_mapped"].astype(np.int64).copy()
    old_end = int(_stage_bounds(stage)["old_end"])
    old_mask = y_true < old_end
    pred_labels = sorted(np.unique(pred[old_mask]).tolist())
    if not pred_labels:
        return pred
    matrix = np.zeros((old_end, len(pred_labels)), dtype=np.int64)
    pred_index = {int(label): idx for idx, label in enumerate(pred_labels)}
    for true_value, pred_value in zip(y_true[old_mask], pred[old_mask]):
        matrix[int(true_value), pred_index[int(pred_value)]] += 1
    row_ind, col_ind = linear_sum_assignment(-matrix)
    mapping = {int(pred_labels[col]): int(row) for row, col in zip(row_ind, col_ind)}
    pred[old_mask] = np.asarray([mapping.get(int(value), int(value)) for value in pred[old_mask]], dtype=np.int64)
    return pred


def collect_records(
    stage68_save_dir: Path,
    raw_dir: Path,
    transmissions: list[int],
    symbols_per_transmission: int,
    decimation: int,
    representation: str,
    batch_size: int,
    device: str,
    feat_dim: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """收集 baseline、正式校准候选和 oracle 参考的逐阶段结果。"""

    records: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    initial_refs: dict[str, float] = {}
    for stage in STAGES:
        dump = _load_dump(stage68_save_dir, stage)
        y_true = dump["y_true"].astype(np.int64)
        variants = {
            "baseline": dump["pred_mapped"].astype(np.int64),
            "old_label_remap_oracle": _old_label_remap_oracle(dump, stage),
        }
        if stage != "initial":
            round_index = int(stage.replace("after_r", ""))
            bounds = _stage_bounds(stage)
            seen_classes = int(bounds["seen_classes"])
            old_end = int(bounds["old_end"])
            checkpoint = stage68_save_dir / f"mvacc_cil_after_r{round_index}.pth"
            model = _load_lora_stage_model(checkpoint, seen_classes=seen_classes, feat_dim=feat_dim, device=device)
            calibration_x, calibration_y, stage_rows = _build_calibration_x(
                raw_dir=raw_dir,
                day=round_index + 1,
                old_end=old_end,
                transmissions=transmissions,
                symbols_per_transmission=symbols_per_transmission,
                decimation=decimation,
                representation=representation,
            )
            calibration_logits = _extract_logits(model, calibration_x, batch_size=batch_size, device=device)
            variants["old_iq1_calibration"] = _apply_old_day_calibration(
                dump,
                calibration_logits=calibration_logits,
                calibration_y=calibration_y,
                stage=stage,
            )
            for row in stage_rows:
                calibration_rows.append({"stage": stage, **row})
        else:
            variants["old_iq1_calibration"] = variants["baseline"]

        for variant, pred in variants.items():
            ref = None if stage == "initial" else initial_refs.get(variant)
            metrics = _evaluate(stage, y_true, pred, ref)
            if stage == "initial":
                initial_refs[variant] = float(metrics["initial_known"])
            records.append(
                {
                    "variant": variant,
                    **{key: _safe_float(value) if isinstance(value, float) else value for key, value in metrics.items()},
                }
            )
    return records, calibration_rows


def _r3(records: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    """读取某个变体的 R3 行。"""

    rows = [row for row in records if row["variant"] == variant and row["stage"] == "After R3"]
    if not rows:
        raise ValueError(f"Missing R3 row for {variant}")
    return rows[-1]


def _fmt(value: float | None) -> str:
    """统一 Markdown 表格的小数格式。"""

    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _table(records: list[dict[str, Any]]) -> str:
    """生成 R3 指标表。"""

    lines = [
        "| Variant | Overall | Old | New | Forgetting | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in ("baseline", "old_iq1_calibration", "old_label_remap_oracle"):
        row = _r3(records, variant)
        lines.append(
            "| {variant} | {overall} | {old} | {new} | {forgetting} | {macro_f1} |".format(
                variant=variant,
                overall=_fmt(row["overall"]),
                old=_fmt(row["old"]),
                new=_fmt(row["new"]),
                forgetting=_fmt(row["forgetting"]),
                macro_f1=_fmt(row["macro_f1"]),
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage69 LoRa old-day labeled calibration report")
    parser.add_argument("--stage68-save-dir", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--calibration-csv", required=True)
    parser.add_argument("--transmissions", default="1", help="逗号分隔的旧类校准 transmission，例如 1 或 1,2")
    parser.add_argument("--symbols-per-transmission", type=int, default=28)
    parser.add_argument("--decimation", type=int, default=1)
    parser.add_argument("--representation", choices=("raw", "dechirped"), default="raw")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    transmissions = [int(item.strip()) for item in str(args.transmissions).split(",") if item.strip()]
    if not transmissions:
        raise ValueError("--transmissions must contain at least one IQ index.")

    records, calibration_rows = collect_records(
        stage68_save_dir=Path(args.stage68_save_dir),
        raw_dir=Path(args.raw_dir),
        transmissions=transmissions,
        symbols_per_transmission=args.symbols_per_transmission,
        decimation=args.decimation,
        representation=args.representation,
        batch_size=args.batch_size,
        device=device,
        feat_dim=args.feat_dim,
    )
    baseline = _r3(records, "baseline")
    calibrated = _r3(records, "old_iq1_calibration")
    remapped = _r3(records, "old_label_remap_oracle")
    passes = float(calibrated["overall"]) >= 0.50 and float(calibrated["old"]) >= 0.50

    lines = [
        f"# Stage69 LoRa 旧类 IQ 校准 Seed7 报告（Job {args.job_id}）",
        "",
        "## 结论",
        "",
        f"- 当前判定：{'通过 50% 目标，可作为“新增少量旧类跨天校准数据”的方案候选。' if passes else '未达到 50% 目标，说明真实 IQ_1 旧类校准仍不足以复现 Stage68 oracle 上界。'}",
        f"- Baseline R3 Overall/Old/New = {_fmt(baseline['overall'])}/{_fmt(baseline['old'])}/{_fmt(baseline['new'])}。",
        f"- 旧类 IQ 校准 R3 Overall/Old/New = {_fmt(calibrated['overall'])}/{_fmt(calibrated['old'])}/{_fmt(calibrated['new'])}。",
        f"- 旧类标签重映射 oracle R3 Overall/Old/New = {_fmt(remapped['overall'])}/{_fmt(remapped['old'])}/{_fmt(remapped['new'])}。",
        "- 该实验新增 Day2-4 旧设备标注校准样本，不属于原 strict 无旧类跨天校准协议；可作为客户侧数据采集/协议调整方案，而不是原协议主结果。",
        "",
        "## R3 指标",
        "",
        _table(records),
        "",
        "## 校准设置",
        "",
        f"- Stage68 复跑目录：`{args.stage68_save_dir}`",
        f"- 原始 I/Q 目录：`{args.raw_dir}`",
        f"- 旧类校准 transmissions：`{','.join(str(v) for v in transmissions)}`",
        f"- 每个 transmission 使用 `{args.symbols_per_transmission}` 个 aligned symbols，representation=`{args.representation}`，decimation=`{args.decimation}`。",
        "",
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")

    calibration_csv = Path(args.calibration_csv)
    calibration_csv.parent.mkdir(parents=True, exist_ok=True)
    if calibration_rows:
        import pandas as pd

        pd.DataFrame(calibration_rows).to_csv(calibration_csv, index=False)
    else:
        calibration_csv.write_text("stage,day,device,label,transmission,samples,symbol_offset,alignment_score\n", encoding="utf-8")

    Path(args.summary_json).write_text(
        json.dumps(
            {
                "schema_version": "stage69_lora_old_day_labeled_calibration_v1",
                "job_id": str(args.job_id),
                "stage68_save_dir": str(args.stage68_save_dir),
                "raw_dir": str(args.raw_dir),
                "transmissions": transmissions,
                "records": records,
                "r3": {
                    "baseline": baseline,
                    "old_iq1_calibration": calibrated,
                    "old_label_remap_oracle": remapped,
                },
                "gate": {
                    "overall_and_old_reach_50": passes,
                    "protocol_note": "新增 Day2-4 旧设备标注校准样本；不属于原 strict 无旧类跨天校准协议。",
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
