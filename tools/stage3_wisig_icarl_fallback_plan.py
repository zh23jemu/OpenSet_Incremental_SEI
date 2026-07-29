"""生成 WiSig RADCIL + iCaRL 低置信回退 seed7 短验证计划。

该计划固定阶段 3 主方法的 MV-ACC 前端与 RADCIL 训练配置，只新增
confidence-gated exemplar fallback：网络分类头先预测，只有低置信样本才
回退到 replay exemplar 原型分类器。阈值只由 Day1 validation 选择。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def build_plan(project_root: Path, dataset_path: str, save_dir: str, job_id: str) -> dict:
    """构造可审计的 seed7 新机制验证计划。"""
    argv = [
        "--dataset_path", dataset_path,
        "--save_dir", save_dir,
        "--selected_rx_list", "2",
        "--initial_known_classes", "10",
        "--round_size", "10",
        "--num_rounds", "3",
        "--seed", "7",
        "--train_closedset",
        "--epochs", "20",
        "--batch_size", "128",
        "--test_batch_size", "256",
        "--disable_visualization",
        "--disable_cil_baselines",
        "--use_supcon",
        "--supcon_weight", "0.1",
        "--cil_head_warmup_epochs", "2",
        "--cil_joint_epochs", "8",
        "--cil_replay_weight", "3.0",
        "--radcil_old_new_batch_ratio", "2.0",
        "--radcil_icarl_fallback",
        "--radcil_icarl_fallback_candidates", "0.45,0.55,0.65,0.75,0.85",
    ]
    return {
        "schema_version": "stage3_wisig_icarl_fallback_seed7_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "stage": "stage3",
        "dataset": "WiSig",
        "job_id": str(job_id),
        "variant": "hybrid_radcil_icarl_exemplar_classifier_fallback",
        "strict_protocol": (
            "Day1 60% train / 10% validation / 30% held-out evaluation; "
            "unknown-round truth is excluded from discovery, fallback calibration and training."
        ),
        "run": {
            "experiment_entry": "experiments/exp_wisig_mvacc_cil_strict.py",
            "dataset_path": dataset_path,
            "save_dir": save_dir,
            "argv": argv,
        },
        "isolated_change": {
            "network_backend": "ratio_2p0_replay_3p0",
            "fallback": "confidence-gated exemplar prototype classifier",
            "threshold_selection": "Day1 validation only; ties prefer lower threshold",
            "description": "网络低置信时离散回退到 replay exemplar 原型预测，不做连续 late-fusion 权重搜索。",
        },
        "acceptance_gate": {
            "baseline_summary": "results/stage2/stage2_wisig_main_single_seed_summary_44433946.json",
            "primary": "R3 Overall Acc 不低于 seed7 MV-ACC-CIL 0.6328",
            "secondary": "R3 Old Acc 提升或 Forgetting 降低；若只提升 New Acc 则不扩三种子。",
            "next_action": "通过后再设计三种子正式验证；未通过则保留为负消融并停止该机制。",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 WiSig iCaRL fallback seed7 短验证计划。")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-path", default="数据集/WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_plan(Path(args.project_root), args.dataset_path, args.save_dir, args.job_id)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"WiSig iCaRL fallback seed7 计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
