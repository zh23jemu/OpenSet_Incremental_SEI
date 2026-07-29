"""生成 LoRa 增量双视图一致性 seed7 预注册计划。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_plan() -> dict:
    """返回固定候选、严格数据边界和预注册判定门槛。"""
    return {
        "stage": "stage6",
        "experiment": "lora_aug_consistency_seed7",
        "dataset": "LoRa RFFP Different Days Indoor compact subset",
        "protocol": "10 known + 5x3 strict; IQ_7 validation; IQ_8-10 held-out evaluation",
        "checkpoint": "results/stage5/lora_representation_screen_seed7_44573249/supcon_rf_aug.pth",
        "fixed_backend": {
            "radcil_old_new_batch_ratio": 2.0,
            "cil_replay_weight": 3.0,
            "mvacc_enforce_round_size": True,
            "discovery_backend": "mvacc",
        },
        "candidates": [
            {"name": "consistency_0p00", "radcil_aug_consistency_weight": 0.0},
            {"name": "consistency_0p05", "radcil_aug_consistency_weight": 0.05},
            {"name": "consistency_0p10", "radcil_aug_consistency_weight": 0.10},
        ],
        "selection_rule": [
            "只使用 IQ_7 R3 Old-Class Acc 选择候选；并列选择较小权重。",
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
            "augmentation_only_on_training_batches": True,
        },
    }


def main() -> int:
    """将计划写入指定 JSON 文件。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/stage6/stage6_lora_aug_consistency_seed7_plan.json",
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
