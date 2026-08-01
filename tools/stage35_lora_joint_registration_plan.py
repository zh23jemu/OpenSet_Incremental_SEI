"""生成 Stage 35 LoRa 联合注册/训练 seed7 矩阵计划。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    """写出固定的三变体计划，避免继续搜索旧的后端小权重。"""

    parser = argparse.ArgumentParser(description="生成 Stage 35 LoRa 计划")
    parser.add_argument(
        "--output",
        default="results/stage35/stage35_lora_joint_registration_plan.json",
    )
    args = parser.parse_args()
    plan = {
        "schema_version": "stage35_lora_joint_registration_plan_v1",
        "seed": 7,
        "mode": "full_cil",
        "fixed": {
            "discovery_backend": "gpcc_recording_consensus",
            "recording_consensus_threshold": 0.65,
            "closedset_backbone": "lora_chirp",
            "cross_day_repr_adaptation": True,
            "lora_ssl_adaptation": True,
            "strict_split": "10 known + 5x3 new classes; IQ_8-10 held-out",
        },
        "variants": [
            {"name": "registration_all", "top_fraction": 1.0},
            {"name": "registration_top0p60", "top_fraction": 0.60},
            {"name": "registration_top0p80", "top_fraction": 0.80},
        ],
        "hypothesis": "高置信样本只用于新类imprint和replay注册，全量伪标签仍进入带权当前轮训练。",
        "gate": {
            "overall": "must exceed Stage34 seed7 0.2476",
            "old_new": "Old and New may not both decrease versus Stage34.",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Stage 35 计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
