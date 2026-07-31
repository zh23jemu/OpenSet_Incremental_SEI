"""LoRa25 Different Days Indoor strict 10+5x3 协议加载器。

本加载器只封装已经对齐好的紧凑 NPZ，不重新下载或重切原始 IQ 文件。
协议边界来自 datasets/lora25_compact/tools/build_lora25_compact.py：

* Day1：10 个已知设备，IQ_1-7 作为开发/训练池，IQ_8-10 作为初始评估；
* Day2/3/4：每轮新增 5 个未知设备，IQ_1-7 只用于 discovery/enrollment；
* 每轮评估使用同一天 IQ_8-10 的累计已见设备，评估标签不返回给发现或选参流程。

这里返回的字典键名与 WiSig/ADS-B strict loader 对齐，方便后续复用
现有实验入口中的 day1_known_train、day2_unknown_round1 等约定。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np


DEFAULT_LORA25_NPZ = (
    Path(__file__).resolve().parent
    / "lora25_compact"
    / "lora25_diffdays_indoor_aligned_group_256.npz"
)


STAGE_KEYS = (
    "day1_known_train",
    "day1_initial_eval",
    "day2_unknown_round1",
    "day2_eval_after_r1",
    "day3_unknown_round2",
    "day3_eval_after_r2",
    "day4_unknown_round3",
    "day4_eval_after_r3",
)


EXPECTED_LABELS = {
    "day1_backbone_train": range(0, 10),
    "day1_known_validation": range(0, 10),
    "day1_known_train": range(0, 10),
    "day1_initial_eval": range(0, 10),
    "day2_unknown_round1": range(10, 15),
    "day2_eval_after_r1": range(0, 15),
    "day3_unknown_round2": range(15, 20),
    "day3_eval_after_r2": range(0, 20),
    "day4_unknown_round3": range(20, 25),
    "day4_eval_after_r3": range(0, 25),
}


EXPECTED_TRANSMISSIONS = {
    "day1_backbone_train": range(1, 7),
    "day1_known_validation": range(7, 8),
    "day1_known_train": range(1, 8),
    "day1_initial_eval": range(8, 11),
    "day2_unknown_round1": range(1, 8),
    "day2_eval_after_r1": range(8, 11),
    "day3_unknown_round2": range(1, 8),
    "day3_eval_after_r2": range(8, 11),
    "day4_unknown_round3": range(1, 8),
    "day4_eval_after_r3": range(8, 11),
}


def _as_path(path: str | Path | None) -> Path:
    """解析 NPZ 路径，保持默认路径可移植。"""
    if path is None:
        return DEFAULT_LORA25_NPZ
    return Path(path).expanduser().resolve()


def _array(npz: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    """读取 NPZ 数组并给出明确错误，避免后续协议审计静默缺字段。"""
    if name not in npz:
        raise KeyError(f"LoRa25 compact NPZ missing key: {name}")
    return np.asarray(npz[name])


def _unique_ints(values: Iterable[int]) -> list[int]:
    """把 numpy 标量序列转成稳定 JSON 友好的 Python int 列表。"""
    return [int(v) for v in np.unique(np.asarray(values, dtype=np.int64))]


def _validate_stage(stage: str, x: np.ndarray, y: np.ndarray, transmissions: np.ndarray) -> None:
    """验证单个 split 的形状、标签和 transmission 范围是否符合 strict 协议。"""
    if x.ndim != 3 or x.shape[1] != 2 or x.shape[2] <= 0:
        raise ValueError(f"{stage} X must have shape [N, 2, L], got {x.shape}")
    if x.dtype != np.float32:
        raise ValueError(f"{stage} X must be float32, got {x.dtype}")
    if y.shape != (x.shape[0],):
        raise ValueError(f"{stage} y length mismatch: {y.shape} vs X={x.shape}")
    if transmissions.shape != (x.shape[0],):
        raise ValueError(f"{stage} transmission length mismatch: {transmissions.shape} vs X={x.shape}")

    expected_labels = list(EXPECTED_LABELS[stage])
    actual_labels = _unique_ints(y)
    if actual_labels != expected_labels:
        raise ValueError(f"{stage} labels mismatch: {actual_labels} != {expected_labels}")

    expected_tx = list(EXPECTED_TRANSMISSIONS[stage])
    actual_tx = _unique_ints(transmissions)
    if actual_tx != expected_tx:
        raise ValueError(f"{stage} transmissions mismatch: {actual_tx} != {expected_tx}")


def _subset_stage(item: dict[str, object], mask: np.ndarray, split: str) -> dict[str, object]:
    """按 transmission 掩码派生 Day1 train/validation，保持记录级边界。"""
    return {
        "X": np.asarray(item["X"])[mask],
        "y": np.asarray(item["y"])[mask],
        "day": item["day"],
        "tx_range": item["tx_range"],
        "split": split,
        "transmission": np.asarray(item["transmission"])[mask],
        "recording_id": np.asarray(item["recording_id"])[mask],
    }


def load_lora25_diffdays_3round(
    npz_path: str | Path | None = None,
    initial_known_classes: int = 10,
    round_size: int = 5,
    num_rounds: int = 3,
) -> Dict[str, object]:
    """加载 LoRa RFFP Different Days Indoor 10 known + 5x3 strict 协议。

    参数保持与其它 strict loader 相似，但当前紧凑数据固定为 25 个设备。
    若调用方请求其它类别规模，直接报错，避免误把 LoRa 紧凑子集当成可任意重切。
    """
    if (initial_known_classes, round_size, num_rounds) != (10, 5, 3):
        raise ValueError("LoRa25 strict loader is specialized for 10 known + 5 x 3 rounds.")

    path = _as_path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"LoRa25 compact NPZ not found: {path}")

    with np.load(path, allow_pickle=False) as npz:
        result: Dict[str, object] = {}
        for stage in STAGE_KEYS:
            x = _array(npz, f"{stage}_X").astype(np.float32, copy=False)
            y = _array(npz, f"{stage}_y").astype(np.int64, copy=False)
            transmissions = _array(npz, f"{stage}_transmission").astype(np.int16, copy=False)
            recording_id = _array(npz, f"{stage}_recording_id").astype(np.int32, copy=False)
            _validate_stage(stage, x, y, transmissions)
            result[stage] = {
                "X": x,
                "y": y,
                "day": stage.split("_", maxsplit=1)[0].replace("day", "Day "),
                "tx_range": f"{int(y.min())}-{int(y.max())}",
                "split": "IQ_1-7 discovery/development" if transmissions.max() <= 7 else "IQ_8-10 held-out evaluation",
                "transmission": transmissions,
                "recording_id": recording_id,
            }

        physical_device_order = _array(npz, "physical_device_order").astype(np.int16, copy=False)
        device_names = _array(npz, "device_names").astype(str, copy=False)
        day_names = _array(npz, "day_names").astype(str, copy=False)

    # Day1 的原紧凑 split 是 70% development pool。这里进一步按完整
    # transmission 固定成 60% backbone train + 10% validation，避免同一
    # 物理发送记录的相邻符号落入训练和验证两侧。
    day1_development = result["day1_known_train"]
    day1_transmissions = np.asarray(day1_development["transmission"])
    backbone_mask = day1_transmissions <= 6
    validation_mask = day1_transmissions == 7
    result["day1_backbone_train"] = _subset_stage(
        day1_development,
        backbone_mask,
        "IQ_1-6 fixed backbone train (60% overall)",
    )
    result["day1_known_validation"] = _subset_stage(
        day1_development,
        validation_mask,
        "IQ_7 fixed validation (10% overall)",
    )
    _validate_stage(
        "day1_backbone_train",
        result["day1_backbone_train"]["X"],
        result["day1_backbone_train"]["y"],
        result["day1_backbone_train"]["transmission"],
    )
    _validate_stage(
        "day1_known_validation",
        result["day1_known_validation"]["X"],
        result["day1_known_validation"]["y"],
        result["day1_known_validation"]["transmission"],
    )

    if _unique_ints(physical_device_order) != list(range(1, 26)):
        raise ValueError("LoRa25 physical_device_order must cover physical Device1..Device25 exactly once.")

    round_counts = []
    for round_index in range(1, num_rounds + 1):
        discovery_key = f"day{round_index + 1}_unknown_round{round_index}"
        eval_key = f"day{round_index + 1}_eval_after_r{round_index}"
        discovery = result[discovery_key]
        evaluation = result[eval_key]
        round_counts.append(
            {
                "round": int(round_index),
                "discovery_samples": int(len(discovery["y"])),
                "evaluation_samples": int(len(evaluation["y"])),
                "new_global_labels": list(EXPECTED_LABELS[discovery_key]),
            }
        )

    result.update(
        {
            "dataset": "LoRa RFFP Different Days Indoor compact aligned subset",
            "npz_path": str(path),
            "protocol_version": "lora25_diffdays_indoor_strict_10known_5x3_v1",
            "initial_known_classes": int(initial_known_classes),
            "round_size": int(round_size),
            "num_rounds": int(num_rounds),
            "sample_length": int(result["day1_known_train"]["X"].shape[2]),
            "physical_device_order": [int(v) for v in physical_device_order.tolist()],
            "device_names": [str(v) for v in device_names.tolist()],
            "day_names": [str(v) for v in day_names.tolist()],
            "round_counts": round_counts,
            "label_policy": "global labels 0..24 follow physical_device_order; Device9 is introduced in R3 because Day2 lacks Device9 source files.",
            "leakage_boundary": "Day1 IQ_1-6 are backbone train, IQ_7 is validation, and IQ_8-10 are held-out evaluation; later-day IQ_1-7 are discovery/enrollment and IQ_8-10 are held-out evaluation.",
            "preprocessing": "LoRa SF7 chirp-boundary alignment, DC removal, per-symbol RMS normalization, decimate by 4.",
        }
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="加载并打印 LoRa25 strict 10+5x3 协议摘要。")
    parser.add_argument("--npz-path", default=str(DEFAULT_LORA25_NPZ))
    args = parser.parse_args()
    splits = load_lora25_diffdays_3round(args.npz_path)
    for key in ("day1_backbone_train", "day1_known_validation", *STAGE_KEYS):
        item = splits[key]
        print(key, item["X"].shape, np.unique(item["y"]).tolist(), np.unique(item["transmission"]).tolist())
