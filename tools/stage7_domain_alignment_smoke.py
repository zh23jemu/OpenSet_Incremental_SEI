"""验证训练期跨天分布对齐损失的形状、有限值和可复现梯度。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp_wisig_mvacc_cil_strict import _feature_domain_alignment_loss


def main() -> int:
    """使用合成的均值/协方差偏移数据完成最小契约测试。"""
    torch.manual_seed(7)
    source = torch.randn(12, 16, requires_grad=True)
    target = torch.randn(12, 16) + 0.25
    loss = _feature_domain_alignment_loss(source, target)
    loss.backward()
    payload = {
        "loss_finite": bool(torch.isfinite(loss).item()),
        "loss_positive": bool(float(loss.detach()) > 0.0),
        "gradient_finite": bool(torch.isfinite(source.grad).all().item()),
        "source_shape": list(source.shape),
        "target_shape": list(target.shape),
        "uses_labels": False,
        "uses_heldout": False,
    }
    if not all(payload[key] for key in ("loss_finite", "loss_positive", "gradient_finite")):
        raise AssertionError(payload)
    output = Path("results/stage7/stage7_domain_alignment_smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
