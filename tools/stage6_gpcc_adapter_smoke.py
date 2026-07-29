"""GPCC 发现前端的合成数据 smoke test。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.graph_prototype_discovery_adapter import run_gpcc


def _make_features(classes: int, samples_per_class: int, dim: int, seed: int) -> dict[str, np.ndarray]:
    """构造三路多视图特征，模拟 deep/RF/graph 视图的轻微差异。"""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 3.0, size=(classes, dim)).astype(np.float32)
    labels = np.repeat(np.arange(classes), samples_per_class)
    base = centers[labels] + rng.normal(0.0, 0.35, size=(len(labels), dim)).astype(np.float32)
    rf = centers[labels] + rng.normal(0.0, 0.55, size=(len(labels), dim)).astype(np.float32)
    graph = centers[labels] + rng.normal(0.0, 0.25, size=(len(labels), dim)).astype(np.float32)
    return {"deep": base, "rf": rf, "graph": graph}


def _check_case(classes: int, seed: int) -> dict:
    """运行单个 K 约束用例并返回可落盘的审计结果。"""
    feats = _make_features(classes=classes, samples_per_class=24, dim=12, seed=seed)
    first = run_gpcc(feats, target_clusters=classes, seed=seed)
    second = run_gpcc(feats, target_clusters=classes, seed=seed)
    unique = sorted(np.unique(first.labels).tolist())
    return {
        "classes": int(classes),
        "samples": int(len(first.labels)),
        "unique_labels": [int(v) for v in unique],
        "cluster_count": int(len(unique)),
        "has_noise": bool(np.any(first.labels < 0)),
        "same_seed_reproducible": bool(np.array_equal(first.labels, second.labels)),
        "confidence_min": float(np.min(first.confidence)),
        "confidence_mean": float(np.mean(first.confidence)),
        "diagnostics": first.diagnostics,
    }


def main() -> int:
    """执行 5 类和 10 类两个硬约束 smoke test。"""
    parser = argparse.ArgumentParser(description="GPCC adapter smoke test")
    parser.add_argument("--output", default="results/stage6/stage6_gpcc_adapter_smoke.json")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cases = [_check_case(5, args.seed), _check_case(10, args.seed)]
    ok = all(
        row["cluster_count"] == row["classes"]
        and not row["has_noise"]
        and row["same_seed_reproducible"]
        for row in cases
    )
    payload = {
        "ok": bool(ok),
        "uses_hdbscan": False,
        "cases": cases,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
