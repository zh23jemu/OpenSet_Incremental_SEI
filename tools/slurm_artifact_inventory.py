"""只读盘点 Slurm 结果目录中的大型二进制和失败日志。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


BINARY_SUFFIXES = {".pth", ".npz"}
LOG_SUFFIXES = {".out", ".err"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output", required=True)
    parser.add_argument("--large-threshold-mb", type=float, default=50.0)
    args = parser.parse_args()

    root = Path(args.results_root)
    threshold = int(float(args.large_threshold_mb) * 1024 * 1024)
    binary_files = []
    nonempty_error_logs = []
    suffix_bytes: Counter[str] = Counter()
    suffix_counts: Counter[str] = Counter()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        size = path.stat().st_size
        suffix_counts[suffix] += 1
        suffix_bytes[suffix] += size
        if suffix in BINARY_SUFFIXES:
            binary_files.append({
                "path": path.as_posix(),
                "suffix": suffix,
                "size_bytes": size,
                "size_mb": round(size / 1024 / 1024, 3),
                "large": size >= threshold,
                "recommended_action": "retain_on_slurm_exclude_from_normal_git",
            })
        elif suffix == ".err" and size > 0:
            nonempty_error_logs.append({
                "path": path.as_posix(),
                "size_bytes": size,
                "recommended_action": "review_then_archive_or_keep",
            })

    binary_files.sort(key=lambda item: (-item["size_bytes"], item["path"]))
    payload = {
        "results_root": root.as_posix(),
        "policy": {
            "destructive_action": "none",
            "normal_git": "small reports only; model/replay binaries remain on Slurm",
            "large_threshold_mb": float(args.large_threshold_mb),
        },
        "binary_summary": {
            "count": len(binary_files),
            "bytes": sum(item["size_bytes"] for item in binary_files),
            "large_count": sum(bool(item["large"]) for item in binary_files),
        },
        "nonempty_error_log_count": len(nonempty_error_logs),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "suffix_bytes": dict(sorted(suffix_bytes.items())),
        "binary_files": binary_files,
        "nonempty_error_logs": nonempty_error_logs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Slurm 产物只读清单：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
