"""生成 LoRa discovery 簇可靠性伪标签加权 seed7 计划。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_plan() -> dict:
    """返回固定二元候选和严格选择门槛。"""
    return {
        "stage": "stage6",
        "experiment": "lora_reliability_weight_seed7",
        "dataset": "LoRa RFFP Different Days Indoor compact subset",
        "protocol": "10 known + 5x3 strict; IQ_7 validation; IQ_8-10 held-out evaluation",
        "fixed_backend": {
            "radcil_old_new_batch_ratio": 2.0,
            "cil_replay_weight": 3.0,
            "mvacc_enforce_round_size": True,
            "pseudo_weight_floor": 0.20,
            "cluster_reliability_floor": 0.20,
        },
        "candidates": [
            {"name": "reliability_off", "use_cluster_reliability": False},
            {"name": "reliability_on", "use_cluster_reliability": True},
        ],
        "selection_rule": [
            "只使用 IQ_7 R3 Old-Class Acc 选择是否采用；并列优先关闭新增机制。",
            "IQ_8-10 只做最终事后审计，不参与选择。",
        ],
        "pass_gate": {
            "validation_old_acc_must_improve": True,
            "heldout_overall_drop_max": 0.01,
            "heldout_new_acc_drop_max": 0.03,
            "heldout_forgetting_must_decrease": True,
        },
        "leakage_policy": {
            "uses_heldout_eval_for_selection": False,
            "uses_ground_truth_for_training": False,
            "reliability_uses_discovery_structure_only": True,
        },
    }


def main() -> int:
    """写出预注册 JSON 计划。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/stage6/stage6_lora_reliability_weight_seed7_plan.json",
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_plan(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已生成计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
