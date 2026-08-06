import argparse
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_lora25_compact import (  # noqa: E402
    DEVICE_ORDER,
    DEVICE_TO_LABEL,
    cache_path,
    stage_specs,
)


def lora_upchirp(n=1024, sample_rate=1_000_000.0, bandwidth=125_000.0):
    t = np.arange(n, dtype=np.float64) / sample_rate
    symbol_time = n / sample_rate
    phase = 2.0 * np.pi * (
        -0.5 * bandwidth * t + 0.5 * bandwidth / symbol_time * t * t
    )
    return np.exp(1j * phase).astype(np.complex64)


def alignment_score(iq, offset, reference, blocks=4):
    n = len(reference)
    values = []
    for block in range(blocks):
        start = int(offset) + block * n
        x = iq[start : start + n]
        if len(x) != n:
            return -np.inf
        spectrum = np.abs(np.fft.fft(x * np.conj(reference)))
        values.append(float(spectrum.max() / (spectrum.sum() + 1e-12)))
    return float(np.mean(values))


def estimate_symbol_offset(iq, reference):
    coarse_offsets = range(0, len(reference), 8)
    coarse_scores = [alignment_score(iq, offset, reference) for offset in coarse_offsets]
    coarse_best = int(list(coarse_offsets)[int(np.argmax(coarse_scores))])
    refine_offsets = [
        offset for offset in range(coarse_best - 8, coarse_best + 9)
        if 0 <= offset < len(reference)
    ]
    refine_scores = [alignment_score(iq, offset, reference) for offset in refine_offsets]
    best_index = int(np.argmax(refine_scores))
    return int(refine_offsets[best_index]), float(refine_scores[best_index])


def aligned_symbols(
    cache_file,
    reference,
    symbols_per_transmission=14,
    decimation=1,
    representation="raw",
    eps=1e-8,
):
    # 原始 LoRa `.dat` 单文件约 153MB；校准和紧凑构建只需要前若干个
    # symbol 片段。使用只读 memmap 可以避免每次把完整文件读入内存，
    # 对 Stage69/70 这类多 transmission 校准矩阵尤其重要，同时保持切片
    # 语义与原来的 ndarray 读取方式一致。
    iq = np.memmap(cache_file, dtype="<c8", mode="r")
    offset, score = estimate_symbol_offset(iq, reference)
    symbols = []
    for index in range(symbols_per_transmission):
        start = offset + index * len(reference)
        symbol = iq[start : start + len(reference)].astype(np.complex64)
        if len(symbol) != len(reference):
            raise ValueError(f"Insufficient samples in {cache_file}")
        # Remove receiver DC and normalize power per independent LoRa symbol.
        symbol = symbol - np.mean(symbol)
        symbol = symbol / np.sqrt(np.mean(np.abs(symbol) ** 2) + eps)
        if representation == "dechirped":
            # Remove the transmitted LoRa symbol index while retaining the
            # fractional carrier-frequency error and analogue impairments.
            # After dechirping, an ideal SF7 symbol is an FFT-bin-centred tone.
            # Shifting by the nearest integer peak therefore suppresses payload
            # content without using a device label.
            symbol = symbol * np.conj(reference)
            peak_bin = int(np.argmax(np.abs(np.fft.fft(symbol))))
            n = np.arange(len(symbol), dtype=np.float32)
            symbol = symbol * np.exp(-2j * np.pi * peak_bin * n / len(symbol))
        symbol = symbol[::decimation]
        symbols.append(np.stack([symbol.real, symbol.imag], axis=0))
    return np.asarray(symbols, dtype=np.float32), offset, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alignment_manifest", required=True)
    parser.add_argument("--symbols_per_transmission", type=int, default=14)
    parser.add_argument(
        "--decimation",
        type=int,
        default=1,
        choices=(1, 2, 4),
        help="Integer downsampling after symbol alignment (1 preserves all 1024 samples).",
    )
    parser.add_argument(
        "--representation",
        choices=("raw", "dechirped"),
        default="raw",
        help="raw aligned chirp or payload-invariant dechirped residual.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output = Path(args.output)
    manifest_path = Path(args.alignment_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    reference = lora_upchirp()

    arrays = {}
    diagnostics = []
    for stage, (day, devices, transmissions) in stage_specs().items():
        stage_x, stage_y, stage_transmission, stage_recording_id = [], [], [], []
        recording_id = 0
        for device in devices:
            for transmission in transmissions:
                record = {
                    "stage": stage,
                    "day": int(day),
                    "device": int(device),
                    "transmission": int(transmission),
                }
                path = cache_path(cache_dir, record)
                if not path.exists():
                    raise FileNotFoundError(path)
                x, offset, score = aligned_symbols(
                    path,
                    reference,
                    symbols_per_transmission=args.symbols_per_transmission,
                    decimation=args.decimation,
                    representation=args.representation,
                )
                stage_x.append(x)
                stage_y.append(
                    np.full(len(x), DEVICE_TO_LABEL[device], dtype=np.int64)
                )
                stage_transmission.append(
                    np.full(len(x), transmission, dtype=np.int16)
                )
                # File/capture identity is observable metadata, not a Tx label.
                # It supports must-link aggregation of symbols from one recorded
                # transmission while retaining every symbol downstream.
                stage_recording_id.append(
                    np.full(len(x), recording_id, dtype=np.int32)
                )
                recording_id += 1
                diagnostics.append({**record, "symbol_offset": offset, "alignment_score": score})
        arrays[f"{stage}_X"] = np.concatenate(stage_x, axis=0)
        arrays[f"{stage}_y"] = np.concatenate(stage_y, axis=0)
        arrays[f"{stage}_transmission"] = np.concatenate(stage_transmission, axis=0)
        arrays[f"{stage}_recording_id"] = np.concatenate(stage_recording_id, axis=0)
        print(
            f"[stage] {stage}: X={arrays[f'{stage}_X'].shape}, "
            f"classes={np.unique(arrays[f'{stage}_y']).size}",
            flush=True,
        )

    arrays["day_names"] = np.asarray(["Day1", "Day2", "Day3", "Day4"])
    arrays["device_names"] = np.asarray([f"Device{i}" for i in DEVICE_ORDER])
    arrays["physical_device_order"] = np.asarray(DEVICE_ORDER, dtype=np.int16)
    np.savez_compressed(output, **arrays)

    scores = np.asarray([item["alignment_score"] for item in diagnostics], dtype=float)
    manifest = {
        "preprocessing": (
            "LoRa SF7 chirp-boundary alignment, DC removal, per-symbol RMS "
            f"normalization, representation={args.representation}, "
            f"decimate by {args.decimation}"
        ),
        "source_sample_rate": 1_000_000,
        "bandwidth": 125_000,
        "source_symbol_samples": 1024,
        "model_symbol_samples": 1024 // args.decimation,
        "decimation": args.decimation,
        "representation": args.representation,
        "symbols_per_transmission": args.symbols_per_transmission,
        "alignment_score_summary": {
            "min": float(scores.min()),
            "mean": float(scores.mean()),
            "max": float(scores.max()),
        },
        "records": diagnostics,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved aligned compact dataset: {output} ({output.stat().st_size} bytes)")
    print(f"Saved alignment manifest: {manifest_path}")


if __name__ == "__main__":
    main()
