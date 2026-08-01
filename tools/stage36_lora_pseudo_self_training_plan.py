"""生成 Stage 36 LoRa 伪标签自训练 seed7 矩阵计划。

本阶段固定发现前端、数据协议和表征适配配置，只验证两个联合训练目标：
当前轮伪类 prototype 归属损失，以及高置信伪标签样本的增强一致性损失。
这样可以把实验重点放在“伪标签如何进入训练”上，避免再次搜索旧的注册比例。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    """写出可审计的三变体实验计划。"""

    parser = argparse.ArgumentParser(description="生成 Stage 36 LoRa 伪标签自训练计划")
    parser.add_argument(
        "--output",
        default="results/stage36/stage36_lora_pseudo_self_training_plan.json",
    )
    args = parser.parse_args()

    plan = {
        "schema_version": "stage36_lora_pseudo_self_training_plan_v1",
        "seed": 7,
        "mode": "full_cil",
        "fixed": {
            "discovery_backend": "gpcc_recording_consensus",
            "recording_consensus_threshold": 0.65,
            "closedset_backbone": "lora_chirp",
            "cross_day_repr_adaptation": True,
            "lora_ssl_adaptation": True,
            "registration_top_fraction": 0.80,
            "strict_split": "10 known + 5x3 new classes; IQ_8-10 held-out",
        },
        "variants": [
            {
                "name": "baseline",
                "new_prototype_weight": 0.0,
                "pseudo_aug_consistency_weight": 0.0,
            },
            {
                "name": "proto0p10_cons0p10",
                "new_prototype_weight": 0.10,
                "pseudo_aug_consistency_weight": 0.10,
            },
            {
                "name": "proto0p20_cons0p20",
                "new_prototype_weight": 0.20,
                "pseudo_aug_consistency_weight": 0.20,
            },
        ],
        "hypothesis": (
            "固定当前轮伪类 prototype 可稳定伪标签归属，增强一致性可减少 "
            "LoRa 跨天扰动下的新类决策抖动；目标是提高 New 和 Overall，"
            "同时不让 Old 明显低于 Stage35 top0.80。"
        ),
        "gate": {
            "overall": "must exceed Stage35 top0.80 Overall 0.2495",
            "new": "must recover to at least Stage34 New 0.4952",
            "old": "must not decrease materially versus Stage35 top0.80 Old 0.1929",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Stage 36 计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
