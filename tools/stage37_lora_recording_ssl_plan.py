"""生成 Stage 37 LoRa recording-level SSL seed7 计划。

Stage 36 证明在伪标签训练阶段追加小 loss 会压低新旧类表现。本阶段把
改动前移到 discovery 前的表征适配：利用可观测 ``recording_id`` 构造
must-link 自监督目标，让同一次 transmission 的多个 symbol 在表征空间内
更一致，同时不读取未知设备真值或 held-out evaluation。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    """写出 Stage 37 seed7 矩阵计划。"""

    parser = argparse.ArgumentParser(description="生成 Stage 37 LoRa recording SSL 计划")
    parser.add_argument(
        "--output",
        default="results/stage37/stage37_lora_recording_ssl_plan.json",
    )
    args = parser.parse_args()
    plan = {
        "schema_version": "stage37_lora_recording_ssl_plan_v1",
        "seed": 7,
        "mode": "full_cil",
        "fixed": {
            "closedset_backbone": "lora_chirp",
            "discovery_backend": "gpcc_recording_consensus",
            "recording_consensus_threshold": 0.65,
            "cross_day_repr_adaptation": True,
            "lora_ssl_adaptation": True,
            "registration_top_fraction": 0.80,
            "strict_split": "10 known + 5x3 new classes; IQ_8-10 held-out",
        },
        "variants": [
            {
                "name": "baseline",
                "recording_ssl": False,
                "recording_weight": 0.0,
                "instance_weight": 0.0,
            },
            {
                "name": "recording_ssl_rec0p50",
                "recording_ssl": True,
                "recording_weight": 0.50,
                "instance_weight": 0.0,
            },
            {
                "name": "recording_ssl_mix0p50",
                "recording_ssl": True,
                "recording_weight": 0.50,
                "instance_weight": 0.20,
            },
        ],
        "gate": {
            "overall": "must exceed Stage36 baseline 0.2552",
            "new": "should not fall below Stage36 baseline New 0.4810",
            "old": "should not materially fall below Stage36 baseline Old 0.1988",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Stage 37 计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
