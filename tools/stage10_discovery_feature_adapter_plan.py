"""生成 Stage 10 discovery 特征适配 seed7 计划。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 Stage 10 discovery 特征适配计划。")
    parser.add_argument(
        "--output",
        default="results/stage10/stage10_discovery_feature_adapter_seed7_plan.json",
    )
    args = parser.parse_args()
    plan = {
        "schema_version": "stage10_discovery_feature_adapter_seed7_v1",
        "stage": "stage10",
        "seed": 7,
        "scope": "discovery-only",
        "datasets": {
            "lora25": {"round_size": 5, "num_rounds": 3},
            "adsb": {"round_size": 10, "num_rounds": 3},
        },
        "backend": "gpcc",
        "adapters": [
            {"name": "none", "smooth_weight": 0.0, "repulsion_weight": 0.0},
            {"name": "mn_smooth", "smooth_weight": 0.20, "repulsion_weight": 0.0},
            {"name": "proto_repulse", "smooth_weight": 0.20, "repulsion_weight": 0.15},
        ],
        "protocol_constraints": [
            "只使用 Day1 known train 和当前轮 discovery 特征",
            "不读取 held-out evaluation 特征或未知真实标签来做决策",
            "每轮输出协议公开的固定簇数",
        ],
        "promotion_gate": {
            "lora": "三轮 mean Hungarian Acc 或 Purity 明显高于当前 GPCC baseline，且无单轮塌缩",
            "adsb": "R3 Hungarian Acc/Purity 高于 target split 或 GPCC 较强 baseline，且簇数不退化",
            "failure_action": "不进入 CIL，归档为 discovery 负消融并转向表征重训",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
