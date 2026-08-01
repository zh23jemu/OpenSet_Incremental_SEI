"""生成 Stage 38 LoRa 物理域自监督预训练计划。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    """写出单候选计划，避免再次做小权重网格搜索。"""

    parser = argparse.ArgumentParser(description="生成 Stage 38 LoRa physical pretrain 计划")
    parser.add_argument("--output", default="results/stage38/stage38_lora_physical_pretrain_plan.json")
    args = parser.parse_args()
    plan = {
        "schema_version": "stage38_lora_physical_pretrain_plan_v1",
        "seed": 7,
        "candidate": "physical_descriptor_pretrain",
        "fixed_protocol": "Day1 IQ_1-6 train, IQ_7 checkpoint selection, IQ_8-10 held-out eval only",
        "physical_targets": [
            "phase_step_mean",
            "phase_step_std",
            "amplitude_mean",
            "amplitude_std",
            "spectrum_centroid",
            "spectrum_bandwidth",
        ],
        "cil_fixed": {
            "closedset_backbone": "lora_chirp",
            "discovery_backend": "gpcc_recording_consensus",
            "recording_consensus_threshold": 0.65,
            "cross_day_repr_adaptation": True,
            "lora_ssl_adaptation": True,
            "registration_top_fraction": 0.80,
        },
        "gate": {
            "overall": "must exceed Stage37 baseline 0.2552",
            "new": "should not drop below 0.4810",
            "old": "should not materially drop below 0.1988",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Stage 38 计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
