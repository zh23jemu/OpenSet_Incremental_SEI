"""Stage82 transmission-symbol pooling 合成 smoke test。

该测试不加载真实 LoRa 数据，只验证三件事：
1. transmission pooling 能把同一组 symbol 聚合并广播回原样本数；
2. 伪标签多数表决输出连续的 group target；
3. 随机 seed 下输出形状稳定，且不产生 ``-1`` noise 标签。
"""

from __future__ import annotations

import numpy as np
import torch

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp_wisig_mvacc_cil_strict import (
    TransmissionAttentionPool,
    _pool_discovery_features_by_transmission,
    _transmission_pooling_losses,
)


def main() -> int:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(20, 16)).astype(np.float32)
    recording_ids = np.repeat(np.arange(5, dtype=np.int64), 4)

    pooled_discovery, diagnostics = _pool_discovery_features_by_transmission(
        features,
        recording_ids,
    )
    assert pooled_discovery.shape == features.shape
    assert diagnostics["Transmission Pooling Groups"] == 5
    assert np.isfinite(pooled_discovery).all()

    pooler = TransmissionAttentionPool(feat_dim=16)
    symbol_features = torch.as_tensor(features)
    pseudo_labels = torch.as_tensor(recording_ids % 5, dtype=torch.long)
    group_tensor = torch.as_tensor(recording_ids, dtype=torch.long)
    pooled, inverse, targets = _transmission_pooling_losses(
        pooler,
        symbol_features,
        pseudo_labels,
        group_tensor,
    )
    assert pooled.shape == (5, 16)
    assert inverse.shape == (20,)
    assert targets.tolist() == [0, 1, 2, 3, 4]
    assert torch.all(targets >= 0)

    print("Stage82 transmission pooling smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
