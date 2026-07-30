"""生成 Stage 8 归一化代理度量 seed7 矩阵计划。

Stage 8 不是继续调旧的 bias、anchor 或 fusion，而是在增量训练阶段
加入归一化 cosine-proxy 度量损失，直接优化旧类 replay 与高置信新类
伪标签的类间角度间隔。计划只跑 seed7，小矩阵不过门槛就不扩三种子。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def weight_tag(weight_text: str) -> str:
    """把权重文本转成稳定目录后缀，例如 0.10 -> 0p10。"""
    return weight_text.strip().replace(".", "p")


def build_plan(args: argparse.Namespace) -> dict:
    """构造 ADS-B 与 LoRa 共用的 seed7 度量学习计划。"""
    weights = [item.strip() for item in args.weights.split(",") if item.strip()]
    return {
        "schema_version": "stage8_incremental_metric_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "stage8",
        "experiment": "incremental_cosine_proxy_metric_seed7",
        "seed": 7,
        "weights": [float(item) for item in weights],
        "weight_items": weights,
        "metric": {
            "loss": "normalized cosine-proxy cross entropy",
            "scale": float(args.metric_scale),
            "margin": float(args.metric_margin),
            "uses_samples": "high-confidence current pseudo labels + replay memory",
            "uses_heldout_for_training": False,
            "uses_unknown_ground_truth": False,
        },
        "datasets": {
            "adsb": {
                "backend": "Long-RADCIL + default target split",
                "baseline_reference": "stage4 target split seed7 weight=0",
                "pass_gate": (
                    "candidate R3 Overall must improve by >=0.01, Old or New must improve by >=0.01, "
                    "and the other of Old/New must not drop by more than 0.02"
                ),
            },
            "lora": {
                "backend": "selected SupCon representation + enforced round-size RADCIL",
                "baseline_reference": "stage7 alignment_off / current selected representation seed7",
                "pass_gate": (
                    "candidate R3 Overall must improve by >=0.02, Old and New must both avoid >0.03 drop, "
                    "and cluster count must stay at 5"
                ),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 Stage 8 增量代理度量 seed7 计划。")
    parser.add_argument("--weights", default="0,0.10,0.25")
    parser.add_argument("--metric-scale", type=float, default=16.0)
    parser.add_argument("--metric-margin", type=float, default=0.05)
    parser.add_argument("--output", default="results/stage8/stage8_incremental_metric_seed7_plan.json")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_plan(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成 Stage 8 计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
