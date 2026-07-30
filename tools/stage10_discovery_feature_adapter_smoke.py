"""Stage 10 discovery 特征适配器 smoke test。

该测试只使用合成特征，验证适配器的工程不变量：输出维度稳定、没有
NaN/Inf、同 seed 输入可复现，并且 proto_repulse 不读取任何未知标签。
真实 ADS-B / LoRa 聚类收益需要通过 discovery-only Slurm 任务判断。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.discovery_feature_adaptation import adapt_discovery_features


def main() -> None:
    rng = np.random.default_rng(7)
    known_centers = rng.normal(size=(4, 16)).astype(np.float32)
    known_x, known_y = [], []
    for class_id, center in enumerate(known_centers):
        known_x.append(center + 0.08 * rng.normal(size=(30, 16)).astype(np.float32))
        known_y.extend([class_id] * 30)
    known_x = np.vstack(known_x).astype(np.float32)
    known_y = np.asarray(known_y, dtype=np.int64)

    novel_centers = rng.normal(size=(3, 16)).astype(np.float32)
    discovery_x = np.vstack([
        center + 0.10 * rng.normal(size=(25, 16)).astype(np.float32)
        for center in novel_centers
    ]).astype(np.float32)

    none = adapt_discovery_features(known_x, known_y, discovery_x, method="none")
    smooth = adapt_discovery_features(known_x, known_y, discovery_x, method="mn_smooth")
    repulse = adapt_discovery_features(
        known_x,
        known_y,
        discovery_x,
        method="proto_repulse",
        smooth_k=8,
        smooth_weight=0.25,
        repulsion_weight=0.20,
    )
    repulse_again = adapt_discovery_features(
        known_x,
        known_y,
        discovery_x,
        method="proto_repulse",
        smooth_k=8,
        smooth_weight=0.25,
        repulsion_weight=0.20,
    )

    outputs = {"none": none, "mn_smooth": smooth, "proto_repulse": repulse}
    checks = {}
    for name, result in outputs.items():
        checks[name] = {
            "shape": list(result.features.shape),
            "finite": bool(np.isfinite(result.features).all()),
            "diagnostics": result.diagnostics,
        }
        if result.features.shape != discovery_x.shape:
            raise AssertionError(f"{name} changed feature shape: {result.features.shape}")
        if not np.isfinite(result.features).all():
            raise AssertionError(f"{name} produced non-finite values.")

    if not np.allclose(repulse.features, repulse_again.features):
        raise AssertionError("proto_repulse is not deterministic for identical inputs.")

    output_path = Path("results/stage10/stage10_discovery_feature_adapter_smoke.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"ok": True, "checks": checks}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
