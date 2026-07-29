"""阶段 6 增量双视图一致性损失的合成 smoke test。

这个测试不加载真实数据，也不读取 held-out evaluation。它只验证新增损失
函数的三个基本不变量，避免把训练期机制的形状错误带到 Slurm 长任务：

1. 同一组特征与自身比较时，余弦一致性损失应接近 0；
2. 方向被改变的增强视图应产生正损失；
3. 特征形状不一致时必须显式报错，而不是静默广播。
"""

from __future__ import annotations

import json
import os
import sys

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.exp_wisig_mvacc_cil_strict import (
    _compute_pseudo_sample_weights,
    _feature_view_consistency_loss,
)


def main() -> None:
    """运行确定性的合成不变量检查并输出 JSON 结果。"""
    torch.manual_seed(7)
    features = torch.randn(8, 16)
    identical = _feature_view_consistency_loss(features, features.clone())

    # 让每个样本的增强视图沿不同方向变化，确保损失不是恒等于 0。
    augmented = features.clone()
    augmented[:, :8] *= -1.0
    changed = _feature_view_consistency_loss(features, augmented)

    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long).numpy()
    probabilities = torch.tensor([1.0, 0.8, 0.9, 0.7]).numpy()
    reliability_weights = _compute_pseudo_sample_weights(
        labels,
        probabilities,
        reliability_map={0: 0.5, 1: 1.0},
        floor=0.2,
        use_cluster_reliability=True,
    )
    reliability_ok = bool(
        reliability_weights[0] < probabilities[0]
        and reliability_weights[2] == probabilities[2]
    )

    shape_error = False
    try:
        _feature_view_consistency_loss(features, torch.randn(7, 16))
    except ValueError:
        shape_error = True

    result = {
        "ok": bool(
            float(identical.item()) < 1e-6
            and float(changed.item()) > 0.1
            and shape_error
            and reliability_ok
        ),
        "same_view_loss": float(identical.item()),
        "changed_view_loss": float(changed.item()),
        "shape_mismatch_rejected": shape_error,
        "cluster_reliability_weighting_ok": reliability_ok,
        "uses_heldout_eval": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
