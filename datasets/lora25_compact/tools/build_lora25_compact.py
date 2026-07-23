import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


BASE_URL = (
    "https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/"
    "LoRa-Dataset/Diff_Days_Indoor_Setup"
)

# Day2 on the official server has no Device9 directory.  Use a deterministic
# order that keeps every class available on its discovery day and all later
# cumulative evaluation days while still using all 25 physical devices.
DEVICE_ORDER = [
    1, 2, 3, 4, 5, 6, 7, 8, 10, 11,
    12, 13, 14, 15, 16,
    17, 18, 19, 20, 21,
    22, 23, 24, 25, 9,
]
DEVICE_TO_LABEL = {device: label for label, device in enumerate(DEVICE_ORDER)}


def stage_specs():
    known = DEVICE_ORDER[:10]
    r1 = DEVICE_ORDER[10:15]
    r2 = DEVICE_ORDER[15:20]
    r3 = DEVICE_ORDER[20:25]
    return {
        "day1_known_train": (1, known, range(1, 8)),
        "day1_initial_eval": (1, known, range(8, 11)),
        "day2_unknown_round1": (2, r1, range(1, 8)),
        "day2_eval_after_r1": (2, known + r1, range(8, 11)),
        "day3_unknown_round2": (3, r2, range(1, 8)),
        "day3_eval_after_r2": (3, known + r1 + r2, range(8, 11)),
        "day4_unknown_round3": (4, r3, range(1, 8)),
        "day4_eval_after_r3": (4, known + r1 + r2 + r3, range(8, 11)),
    }


def make_records():
    records = []
    for stage, (day, devices, transmissions) in stage_specs().items():
        for device in devices:
            for transmission in transmissions:
                records.append(
                    {
                        "stage": stage,
                        "day": int(day),
                        "device": int(device),
                        "transmission": int(transmission),
                    }
                )
    return records


def source_url(record):
    return (
        f"{BASE_URL}/Day{record['day']}/Device{record['device']}/"
        f"IQ_{record['transmission']}.dat"
    )


def cache_path(cache_dir, record):
    return cache_dir / (
        f"day{record['day']}_device{record['device']:02d}_"
        f"transmission{record['transmission']:02d}.bin"
    )


def download_range(record, cache_dir, start_sample, span_samples, retries=6):
    path = cache_path(cache_dir, record)
    expected_bytes = int(span_samples) * 8  # cf32 = complex float32
    byte_start = int(start_sample) * 8
    byte_end = byte_start + expected_bytes - 1
    if path.exists() and path.stat().st_size == expected_bytes:
        payload = path.read_bytes()
    else:
        url = source_url(record)
        last_error = None
        for attempt in range(retries):
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "Range": f"bytes={byte_start}-{byte_end}",
                        "User-Agent": "OpenSet-Incremental-SEI-research/1.0",
                    },
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = response.read()
                    status = getattr(response, "status", None)
                if status != 206:
                    raise RuntimeError(f"Expected HTTP 206, got {status}: {url}")
                if len(payload) != expected_bytes:
                    raise RuntimeError(
                        f"Range size mismatch for {url}: {len(payload)} != {expected_bytes}"
                    )
                path.write_bytes(payload)
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 == retries:
                    raise
                status_code = getattr(exc, "code", None)
                if status_code == 403:
                    # The public Apache server temporarily rate-limits bursts.
                    time.sleep(15 * (attempt + 1))
                else:
                    time.sleep(2 ** attempt)
        else:
            raise RuntimeError(last_error)

    return {
        **record,
        "url": source_url(record),
        "cache_file": str(path.resolve()),
        "byte_start": byte_start,
        "byte_end": byte_end,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def payload_to_windows(path, windows_per_transmission, window_len, stride):
    iq = np.fromfile(path, dtype="<c8")
    starts = np.arange(windows_per_transmission, dtype=np.int64) * int(stride)
    if int(starts[-1]) + window_len > len(iq):
        raise ValueError(f"Cached range is too short: {path}")
    windows = np.stack([iq[s : s + window_len] for s in starts], axis=0)
    windows = np.stack([windows.real, windows.imag], axis=1).astype(np.float32)
    return np.nan_to_num(windows, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--start_sample", type=int, default=2_000_000)
    parser.add_argument("--windows_per_transmission", type=int, default=16)
    parser.add_argument("--window_len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="Download only the first N records for a probe.")
    args = parser.parse_args()

    output = Path(args.output)
    cache_dir = Path(args.cache_dir)
    manifest_path = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    span_samples = (
        (args.windows_per_transmission - 1) * args.stride + args.window_len
    )
    records = make_records()
    if args.limit is not None:
        records = records[: args.limit]

    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_range,
                record,
                cache_dir,
                args.start_sample,
                span_samples,
            ): record
            for record in records
        }
        for index, future in enumerate(as_completed(futures), start=1):
            completed.append(future.result())
            if index == 1 or index % 10 == 0 or index == len(futures):
                print(f"[download] {index}/{len(futures)} ranges complete", flush=True)

    completed.sort(key=lambda x: (x["day"], x["device"], x["transmission"]))
    manifest = {
        "dataset": "Oregon State LoRa Diff Days Indoor",
        "base_url": BASE_URL,
        "source_datatype": "cf32 little-endian",
        "start_sample": args.start_sample,
        "span_samples": span_samples,
        "window_len": args.window_len,
        "stride": args.stride,
        "windows_per_transmission": args.windows_per_transmission,
        "mapped_label_to_physical_device": {
            str(label): int(device) for label, device in enumerate(DEVICE_ORDER)
        },
        "protocol": {
            "initial": "Day1 physical Device1-8,10,11; IQ_1-7 train, IQ_8-10 eval",
            "r1": "Day2 physical Device12-16 discovery; cumulative eval",
            "r2": "Day3 physical Device17-21 discovery; cumulative eval",
            "r3": "Day4 physical Device22-25,9 discovery; cumulative eval",
            "missing_source_note": "Official Day2 directory has no Device9; Device9 is introduced in R3.",
        },
        "records": completed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.limit is not None:
        print(f"Probe completed; manifest={manifest_path}")
        return

    arrays = {}
    record_map = {
        (r["stage"], r["day"], r["device"], r["transmission"]): r
        for r in completed
    }
    for stage, (day, devices, transmissions) in stage_specs().items():
        stage_x, stage_y, stage_transmission = [], [], []
        for device in devices:
            for transmission in transmissions:
                record = record_map[(stage, day, device, transmission)]
                x = payload_to_windows(
                    record["cache_file"],
                    args.windows_per_transmission,
                    args.window_len,
                    args.stride,
                )
                stage_x.append(x)
                stage_y.append(
                    np.full(len(x), DEVICE_TO_LABEL[device], dtype=np.int64)
                )
                stage_transmission.append(
                    np.full(len(x), transmission, dtype=np.int16)
                )
        arrays[f"{stage}_X"] = np.concatenate(stage_x, axis=0)
        arrays[f"{stage}_y"] = np.concatenate(stage_y, axis=0)
        arrays[f"{stage}_transmission"] = np.concatenate(stage_transmission, axis=0)
        print(
            f"[stage] {stage}: X={arrays[f'{stage}_X'].shape}, "
            f"classes={np.unique(arrays[f'{stage}_y']).size}",
            flush=True,
        )

    arrays["day_names"] = np.asarray(["Day1", "Day2", "Day3", "Day4"])
    arrays["device_names"] = np.asarray([f"Device{i}" for i in DEVICE_ORDER])
    arrays["physical_device_order"] = np.asarray(DEVICE_ORDER, dtype=np.int16)
    np.savez_compressed(output, **arrays)
    print(f"Saved compact dataset: {output} ({output.stat().st_size} bytes)")
    print(f"Saved source manifest: {manifest_path}")


if __name__ == "__main__":
    main()
