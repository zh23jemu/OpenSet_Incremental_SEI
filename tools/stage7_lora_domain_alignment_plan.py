"""生成 LoRa 训练期跨天特征分布对齐的 seed7 预注册计划。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_plan() -> dict:
    """返回只比较关闭/开启的结构性候选和严格选择门槛。"""
    return {
        "stage": "stage7",
        "experiment": "lora_domain_alignment_seed7",
        "dataset": "LoRa RFFP Different Days Indoor compact subset",
        "protocol": "10 known + 5x3 strict; IQ_7 validation; IQ_8-10 held-out evaluation",
        "mechanism": {
            "name": "training-time cross-day CORAL-style alignment",
            "source": "current discovery features",
            "target": "historical replay features",
            "statistics": "normalized feature mean and covariance",
            "uses_heldout_for_training": False,
            "uses_unknown_ground_truth": False,
        },
        "fixed_backend": {
            "radcil_old_new_batch_ratio": 2.0,
            "cil_replay_weight": 3.0,
            "mvacc_enforce_round_size": True,
            "representation_checkpoint": "results/stage5/lora_representation_screen_seed7_44573249/supcon_rf_aug.pth",
        },
        "candidates": [
            {"name": "alignment_off", "weight": 0.0},
            {"name": "alignment_on", "weight": 0.05},
        ],
        "selection_rule": [
            "只使用 IQ_7 R3 Old-Class Acc 选择是否采用。",
            "IQ_8-10 只做最终事后审计，不参与选参。",
            "候选必须提升 IQ_7 旧类保持，并满足 held-out Overall 不下降超过 0.01、New 不下降超过 0.03、Forgetting 严格下降。",
        ],
        "pass_gate": {
            "validation_old_acc_must_improve": True,
            "heldout_overall_drop_max": 0.01,
            "heldout_new_acc_drop_max": 0.03,
            "heldout_forgetting_must_decrease": True,
        },
    }


def main() -> int:
    """写出计划 JSON，供 Slurm 入口和报告器共同引用。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results/stage7/stage7_lora_domain_alignment_seed7_plan.json",
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_plan(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
