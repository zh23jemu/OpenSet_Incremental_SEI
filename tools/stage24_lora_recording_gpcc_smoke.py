"""Stage 24 recording-level GPCC 合成 smoke test。

该测试只验证算法契约：固定 K、无 noise、同一 recording 内标签一致、
同 seed 可复现。合成真值只用于构造可分样本，不会传给聚类器。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.recording_consensus_discovery_adapter import run_recording_gpcc


def _make_features(
    *,
    classes: int,
    recordings_per_class: int,
    symbols_per_recording: int,
    feature_dim: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """构造包含 recording 内噪声的三视图合成特征。"""

    rng = np.random.default_rng(int(seed))
    views = {"deep": [], "rf": [], "graph": []}
    recording_ids = []
    recording_id = 0
    centers = rng.normal(size=(int(classes), int(feature_dim))).astype(np.float32)
    for class_id in range(int(classes)):
        for _ in range(int(recordings_per_class)):
            recording_shift = rng.normal(scale=0.12, size=feature_dim).astype(np.float32)
            for view_name, view_scale in (("deep", 0.10), ("rf", 0.18), ("graph", 0.08)):
                symbols = (
                    centers[class_id]
                    + recording_shift
                    + rng.normal(
                        scale=view_scale,
                        size=(int(symbols_per_recording), int(feature_dim)),
                    ).astype(np.float32)
                )
                views[view_name].append(symbols)
            recording_ids.extend([recording_id] * int(symbols_per_recording))
            recording_id += 1
    return (
        {
            name: np.concatenate(chunks, axis=0).astype(np.float32)
            for name, chunks in views.items()
        },
        np.asarray(recording_ids, dtype=np.int32),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 24 recording-level GPCC 合成 smoke")
    parser.add_argument("--output", default="results/stage24/stage24_lora_recording_gpcc_smoke.json")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    classes = 5
    symbols_per_recording = 7
    features, recording_ids = _make_features(
        classes=classes,
        recordings_per_class=3,
        symbols_per_recording=symbols_per_recording,
        feature_dim=12,
        seed=args.seed,
    )
    first = run_recording_gpcc(features, recording_ids, target_clusters=classes, seed=args.seed)
    second = run_recording_gpcc(features, recording_ids, target_clusters=classes, seed=args.seed)

    recording_consistent = all(
        np.unique(first.labels[recording_ids == recording_id]).size == 1
        for recording_id in np.unique(recording_ids)
    )
    result = {
        "schema_version": "stage24_recording_gpcc_smoke_v1",
        "classes": classes,
        "samples": int(recording_ids.size),
        "recordings": int(np.unique(recording_ids).size),
        "fixed_cluster_count": int(np.unique(first.labels).size),
        "no_noise": bool(np.all(first.labels >= 0)),
        "recording_consistent": bool(recording_consistent),
        "deterministic": bool(
            np.array_equal(first.labels, second.labels)
            and np.allclose(first.confidence, second.confidence)
        ),
        "diagnostics": first.diagnostics,
    }
    result["ok"] = bool(
        result["fixed_cluster_count"] == classes
        and result["no_noise"]
        and result["recording_consistent"]
        and result["deterministic"]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not result["ok"]:
        raise RuntimeError(f"Stage 24 smoke failed: {result}")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
