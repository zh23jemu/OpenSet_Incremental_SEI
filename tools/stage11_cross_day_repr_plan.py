"""生成 Stage 11 跨天表征适配 discovery-only 计划。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 Stage 11 跨天表征适配计划。")
    parser.add_argument(
        "--output",
        default="results/stage11/stage11_cross_day_repr_seed7_plan.json",
    )
    args = parser.parse_args()

    plan = {
        "schema_version": "stage11_cross_day_repr_seed7_v1",
        "stage": "stage11",
        "seed": 7,
        "scope": "discovery-only",
        "motivation": "把跨天一致性从增量后端前移到 discovery 前的训练期表征适配",
        "datasets": {
            "lora25": {"round_size": 5, "num_rounds": 3},
            "adsb": {"round_size": 10, "num_rounds": 3},
        },
        "backend": "gpcc",
        "methods": [
            {"name": "none", "epochs": 0},
            {
                "name": "cross_day",
                "epochs": 2,
                "lr": 1e-5,
                "consistency_weight": 0.5,
                "coral_weight": 0.05,
            },
        ],
        "protocol_constraints": [
            "适配阶段只使用 Day1 known train 标签和当前轮 discovery 无标签样本",
            "不读取 held-out evaluation 数据，也不使用当前轮 discovery 真实标签做决策",
            "每轮使用协议公开的 round_size 作为 GPCC 目标簇数",
            "none 与 cross_day 使用同一 checkpoint、同一 GPCC 参数和同一 seed",
        ],
        "promotion_gate": {
            "lora": "cross_day 三轮 mean Hungarian Acc 或 Purity 明显高于 none，且 R3 不塌缩",
            "adsb": "cross_day R3 Hungarian Acc/Purity 高于 none，且三轮固定簇数不退化",
            "failure_action": "不过门槛则归档为表征适配负消融，不进入 CIL，不扩 seed13/31",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
