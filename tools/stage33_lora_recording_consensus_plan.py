"""生成 Stage 33 LoRa recording-consensus discovery-only 实验计划。

本阶段只验证聚类前端，不进入增量训练。普通 GPCC 作为同一 Chirp/
cross-day/SSL 条件下的基线，recording-consensus 则只在 symbol 级聚类后，
对组内多数标签达到指定比例的 recording 做统一修正。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_plan() -> dict[str, object]:
    """返回固定的 seed7 变体矩阵，避免把验证集或未知真值带入选参。"""

    return {
        "schema_version": "stage33_lora_recording_consensus_plan_v1",
        "dataset": "lora25_diffdays_indoor",
        "seed": 7,
        "mode": "discovery_only",
        "fixed_conditions": {
            "closedset_backbone": "lora_chirp",
            "discovery_backend_base": "gpcc",
            "cross_day_repr_adaptation": True,
            "lora_ssl_adaptation": True,
            "strict_split": "10 known + 5 new classes x 3 rounds; IQ_8-10 held-out",
        },
        "variants": [
            {
                "name": "gpcc",
                "discovery_backend": "gpcc",
                "recording_consensus_threshold": None,
            },
            {
                "name": "consensus_t0p65",
                "discovery_backend": "gpcc_recording_consensus",
                "recording_consensus_threshold": 0.65,
            },
            {
                "name": "consensus_t0p75",
                "discovery_backend": "gpcc_recording_consensus",
                "recording_consensus_threshold": 0.75,
            },
            {
                "name": "consensus_t0p85",
                "discovery_backend": "gpcc_recording_consensus",
                "recording_consensus_threshold": 0.85,
            },
        ],
        "gate": {
            "fixed_target_cluster_count": True,
            "zero_noise": True,
            "relative_to_gpcc": "ARI or Hungarian must improve by > 0.01; the other may not fall by more than 0.01.",
        },
    }


def main() -> int:
    """写出 JSON 计划，供 Slurm 入口和人工审计共同使用。"""

    parser = argparse.ArgumentParser(description="生成 Stage 33 LoRa 计划")
    parser.add_argument(
        "--output",
        default="results/stage33/stage33_lora_recording_consensus_plan.json",
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_plan(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Stage 33 计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
