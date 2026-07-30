"""阶段 8 归一化代理度量损失的本地不变量测试。

测试只使用合成张量，验证：
1. 正常输入可以反向传播；
2. 空 batch 和无效标签返回零损失；
3. 输入维度、缩放系数和 margin 的边界检查生效；
4. 同一随机种子下结果可复现。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.incremental_metric_learning import cosine_proxy_metric_loss


def main() -> None:
    torch.manual_seed(7)
    features = torch.randn(12, 8, requires_grad=True)
    labels = torch.arange(12, dtype=torch.long) % 4
    weights = torch.randn(4, 8, requires_grad=True)

    loss = cosine_proxy_metric_loss(features, labels, weights, scale=16.0, margin=0.05)
    loss.backward()
    if not torch.isfinite(loss) or features.grad is None or weights.grad is None:
        raise AssertionError("正常代理度量损失没有产生有效梯度。")

    empty = cosine_proxy_metric_loss(features[:0], labels[:0], weights)
    if float(empty.detach()) != 0.0:
        raise AssertionError("空 batch 没有返回零损失。")

    invalid = cosine_proxy_metric_loss(features, torch.full_like(labels, -1), weights)
    if float(invalid.detach()) != 0.0:
        raise AssertionError("无效标签没有返回零损失。")

    try:
        cosine_proxy_metric_loss(features, labels, weights, scale=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("scale=0 没有触发参数检查。")

    torch.manual_seed(7)
    features_a = torch.randn(6, 8)
    weights_a = torch.randn(4, 8)
    result_a = cosine_proxy_metric_loss(features_a, labels[:6], weights_a)
    torch.manual_seed(7)
    features_b = torch.randn(6, 8)
    weights_b = torch.randn(4, 8)
    result_b = cosine_proxy_metric_loss(features_b, labels[:6], weights_b)
    if not torch.allclose(result_a, result_b):
        raise AssertionError("相同随机种子下代理度量损失不可复现。")

    output = {
        "ok": True,
        "loss": float(loss.detach()),
        "empty_loss": float(empty.detach()),
        "invalid_label_loss": float(invalid.detach()),
        "reproducible": True,
    }
    output_path = Path("results/stage8/stage8_incremental_metric_smoke.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
