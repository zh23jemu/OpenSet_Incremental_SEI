"""Leakage-aware loader for the official ORACLE OTA RF fingerprinting dataset.

The official GENESYS release stores each recording as a SigMF pair:
``*.sigmf-meta`` + ``*.sigmf-data``.  The metadata advertises cf32, but the
official release page explicitly states that the binary payload must be parsed
as complex128.  This loader therefore defaults to little-endian complex128.

The default confirmatory protocol uses one distance for enrollment/discovery
and a different distance for evaluation.  Consequently, no IQ window can occur
in both sets.  Device labels are used only to construct the staged benchmark;
they are not exposed to the clustering algorithm.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_FILENAME_RE = re.compile(
    r"WiFi_air_X310_(?P<serial>[^_]+)_(?P<distance>\d+)ft_run(?P<run>\d+)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class OracleRecording:
    serial: str
    distance_ft: int
    run: int
    data_path: Path
    meta_path: Path
    sample_rate: float
    sample_count: int


def _metadata_value(obj: dict, key: str, default=None):
    global_obj = obj.get("global", obj.get("core:global", {}))
    return global_obj.get(key, global_obj.get(f"core:{key}", default))


def _parse_identity(meta_path: Path, meta: dict) -> Tuple[str, int, int]:
    match = _FILENAME_RE.search(meta_path.name)
    if match:
        return (
            match.group("serial"),
            int(match.group("distance")),
            int(match.group("run")),
        )

    annotations = meta.get("annotations", [])
    for item in annotations:
        tx = item.get("genesys:transmitter_identification", item.get("transmitter_identification", {}))
        serial = tx.get("serial_number") or tx.get("genesys:serial_number")
        distance = item.get("genesys:distance", item.get("distance"))
        if serial and distance is not None:
            distance_match = re.search(r"\d+", str(distance))
            if distance_match:
                run_match = re.search(r"run(\d+)", meta_path.name, flags=re.IGNORECASE)
                return str(serial), int(distance_match.group()), int(run_match.group(1)) if run_match else 1

    raise ValueError(f"Cannot parse transmitter serial/distance from {meta_path}")


def scan_oracle_recordings(dataset_root: str, dtype: str = "<c16") -> List[OracleRecording]:
    """Recursively scan an extracted official ORACLE Dataset #1 directory."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"ORACLE dataset root does not exist: {root}")

    bytes_per_sample = np.dtype(dtype).itemsize
    rows: List[OracleRecording] = []
    for meta_path in sorted(root.rglob("*.sigmf-meta")):
        data_path = meta_path.with_suffix("").with_suffix(".sigmf-data")
        if not data_path.exists():
            # Standard SigMF names replace only the final suffix.
            data_path = Path(str(meta_path)[: -len(".sigmf-meta")] + ".sigmf-data")
        if not data_path.exists():
            continue

        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        serial, distance_ft, run = _parse_identity(meta_path, meta)
        size = data_path.stat().st_size
        if size % bytes_per_sample != 0:
            raise ValueError(
                f"Unexpected binary size for complex128 payload: {data_path} "
                f"({size} bytes is not divisible by {bytes_per_sample})"
            )
        rows.append(
            OracleRecording(
                serial=serial,
                distance_ft=distance_ft,
                run=run,
                data_path=data_path,
                meta_path=meta_path,
                sample_rate=float(_metadata_value(meta, "sample_rate", 5_000_000.0)),
                sample_count=size // bytes_per_sample,
            )
        )

    if not rows:
        raise FileNotFoundError(
            f"No complete .sigmf-meta/.sigmf-data pairs were found under {root}"
        )
    return rows


def _recording_index(recordings: Sequence[OracleRecording]) -> Dict[Tuple[str, int], List[OracleRecording]]:
    index: Dict[Tuple[str, int], List[OracleRecording]] = {}
    for recording in recordings:
        index.setdefault((recording.serial, recording.distance_ft), []).append(recording)
    for values in index.values():
        values.sort(key=lambda item: (item.run, str(item.data_path)))
    return index


def _common_distances(recordings: Sequence[OracleRecording], serials: Sequence[str]) -> List[int]:
    per_serial = []
    for serial in serials:
        per_serial.append({row.distance_ft for row in recordings if row.serial == serial})
    if not per_serial:
        return []
    return sorted(set.intersection(*per_serial))


def _window_candidates(
    recording: OracleRecording,
    window_length: int,
    max_candidates: int,
    dtype: str,
) -> Tuple[np.memmap, np.ndarray]:
    iq = np.memmap(recording.data_path, dtype=np.dtype(dtype), mode="r")
    usable = (len(iq) // window_length) * window_length
    if usable < window_length:
        raise ValueError(f"Recording is shorter than one window: {recording.data_path}")
    n_windows = usable // window_length
    take = min(int(max_candidates), int(n_windows))
    ids = np.linspace(0, n_windows - 1, num=take, dtype=np.int64)
    return iq, ids * int(window_length)


def _sample_windows(
    recordings: Sequence[OracleRecording],
    count: int,
    window_length: int,
    seed: int,
    dtype: str = "<c16",
    active_quantile: float = 0.50,
    rms_normalize: bool = True,
    max_candidates_per_recording: int = 12000,
) -> np.ndarray:
    if not recordings:
        raise ValueError("No recordings supplied for window extraction")
    if not 0.0 <= active_quantile < 1.0:
        raise ValueError("active_quantile must be in [0, 1)")

    rng = np.random.default_rng(seed)
    candidates: List[Tuple[OracleRecording, int, float]] = []
    for recording in recordings:
        iq, starts = _window_candidates(
            recording, window_length, max_candidates_per_recording, dtype
        )
        energies = np.empty(len(starts), dtype=np.float64)
        batch = 512
        for begin in range(0, len(starts), batch):
            chunk = starts[begin : begin + batch]
            for j, start in enumerate(chunk, start=begin):
                values = np.asarray(iq[start : start + window_length])
                energies[j] = float(np.mean(values.real ** 2 + values.imag ** 2))
        finite = np.isfinite(energies)
        threshold = float(np.quantile(energies[finite], active_quantile)) if np.any(finite) else 0.0
        for start, energy in zip(starts[finite], energies[finite]):
            if energy >= threshold and energy > 0.0:
                candidates.append((recording, int(start), float(energy)))

    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} active non-overlapping windows are available; requested {count}"
        )

    chosen = rng.choice(len(candidates), size=int(count), replace=False)
    output = np.empty((count, 2, window_length), dtype=np.float32)
    memmaps: Dict[Path, np.memmap] = {}
    for out_i, candidate_i in enumerate(chosen):
        recording, start, _ = candidates[int(candidate_i)]
        if recording.data_path not in memmaps:
            memmaps[recording.data_path] = np.memmap(
                recording.data_path, dtype=np.dtype(dtype), mode="r"
            )
        values = np.asarray(
            memmaps[recording.data_path][start : start + window_length], dtype=np.complex128
        )
        if rms_normalize:
            rms = float(np.sqrt(np.mean(values.real ** 2 + values.imag ** 2)))
            if rms > 1e-12:
                values = values / rms
        output[out_i, 0] = values.real.astype(np.float32)
        output[out_i, 1] = values.imag.astype(np.float32)
    return output


def load_oracle_4known_3round(
    dataset_root: str,
    discovery_distance_ft: Optional[int] = None,
    evaluation_distance_ft: Optional[int] = None,
    initial_known_classes: int = 4,
    round_size: int = 4,
    num_rounds: int = 3,
    train_samples_per_class: int = 300,
    eval_samples_per_class: int = 120,
    window_length: int = 256,
    seed: int = 2026,
    class_order_seed: int = 2026,
    dtype: str = "<c16",
    active_quantile: float = 0.50,
    rms_normalize: bool = True,
) -> dict:
    """Create a 4 -> 8 -> 12 -> 16 staged unknown-only benchmark.

    Discovery/enrollment windows and evaluation windows come from different
    distance recordings.  This is stricter than randomly splitting adjacent
    windows from one continuous recording.
    """
    expected_classes = int(initial_known_classes + num_rounds * round_size)
    recordings = scan_oracle_recordings(dataset_root, dtype=dtype)
    serials = sorted({item.serial for item in recordings})
    if len(serials) < expected_classes:
        raise ValueError(
            f"Protocol needs {expected_classes} transmitters, but only {len(serials)} were found"
        )

    class_rng = np.random.default_rng(class_order_seed)
    serials = [serials[i] for i in class_rng.permutation(len(serials))[:expected_classes]]
    distances = _common_distances(recordings, serials)
    if len(distances) < 2:
        raise ValueError(
            "At least two distances common to all selected transmitters are required "
            "for recording-disjoint discovery/evaluation"
        )
    if discovery_distance_ft is None:
        discovery_distance_ft = distances[0]
    if evaluation_distance_ft is None:
        evaluation_distance_ft = distances[1]
    if discovery_distance_ft == evaluation_distance_ft:
        raise ValueError("Discovery and evaluation distances must differ")
    if discovery_distance_ft not in distances or evaluation_distance_ft not in distances:
        raise ValueError(
            f"Requested distances are not common to all devices; available common distances={distances}"
        )

    index = _recording_index(recordings)
    label_of = {serial: label for label, serial in enumerate(serials)}

    def collect(selected_serials: Sequence[str], distance_ft: int, count: int, salt: int):
        xs, ys = [], []
        for serial in selected_serials:
            rows = index.get((serial, int(distance_ft)), [])
            x = _sample_windows(
                rows,
                count=count,
                window_length=window_length,
                seed=seed + salt + 1009 * label_of[serial],
                dtype=dtype,
                active_quantile=active_quantile,
                rms_normalize=rms_normalize,
            )
            xs.append(x)
            ys.append(np.full(count, label_of[serial], dtype=np.int64))
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    groups: List[List[str]] = []
    groups.append(serials[:initial_known_classes])
    cursor = initial_known_classes
    for _ in range(num_rounds):
        groups.append(serials[cursor : cursor + round_size])
        cursor += round_size

    x_train, y_train = collect(groups[0], discovery_distance_ft, train_samples_per_class, 11)
    x_initial_eval, y_initial_eval = collect(groups[0], evaluation_distance_ft, eval_samples_per_class, 21)

    result = {
        "day1_known_train": {
            "X": x_train,
            "y": y_train,
            "day": f"{discovery_distance_ft}ft discovery condition",
            "tx_range": "0-3",
            "split": "recording-disjoint",
        },
        "day1_initial_eval": {
            "X": x_initial_eval,
            "y": y_initial_eval,
            "day": f"{evaluation_distance_ft}ft evaluation condition",
            "tx_range": "0-3",
            "split": "recording-disjoint",
        },
    }

    seen: List[str] = list(groups[0])
    for round_index in range(1, num_rounds + 1):
        new_serials = groups[round_index]
        seen.extend(new_serials)
        x_discovery, y_discovery = collect(
            new_serials, discovery_distance_ft, train_samples_per_class, 100 + round_index
        )
        x_eval, y_eval = collect(
            seen, evaluation_distance_ft, eval_samples_per_class, 200 + round_index
        )
        result[f"day{round_index + 1}_unknown_round{round_index}"] = {
            "X": x_discovery,
            "y": y_discovery,
            "day": f"{discovery_distance_ft}ft discovery condition",
            "tx_range": f"{initial_known_classes + (round_index - 1) * round_size}-"
            f"{initial_known_classes + round_index * round_size - 1}",
            "split": "recording-disjoint",
        }
        result[f"day{round_index + 1}_eval_after_r{round_index}"] = {
            "X": x_eval,
            "y": y_eval,
            "day": f"{evaluation_distance_ft}ft evaluation condition",
            "tx_range": f"0-{initial_known_classes + round_index * round_size - 1}",
            "split": "recording-disjoint",
        }

    result.update(
        {
            "dataset": "ORACLE Dataset #1 OTA raw IQ",
            "serial_order": serials,
            "label_map": label_of,
            "discovery_distance_ft": int(discovery_distance_ft),
            "evaluation_distance_ft": int(evaluation_distance_ft),
            "window_length": int(window_length),
            "dtype": dtype,
            "rms_normalize": bool(rms_normalize),
            "common_distances_ft": distances,
        }
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument("--discovery_distance_ft", type=int, default=None)
    parser.add_argument("--evaluation_distance_ft", type=int, default=None)
    args = parser.parse_args()
    splits = load_oracle_4known_3round(
        args.dataset_root,
        discovery_distance_ft=args.discovery_distance_ft,
        evaluation_distance_ft=args.evaluation_distance_ft,
    )
    for name, item in splits.items():
        if isinstance(item, dict) and "X" in item:
            print(name, item["X"].shape, np.unique(item["y"]).tolist(), item["day"])
