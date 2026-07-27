"""生成 WiSig hybrid RADCIL + DOI-memory seed 7 短验证计划。

该计划固定使用阶段 3 主方法的 MV-ACC 前端和 RADCIL 训练配置，仅新增
DOI-style replay 原型对齐及后验 late fusion，避免同时改变多个实验因素。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def build_plan(project_root: Path, dataset_path: str, save_dir: str, job_id: str) -> dict:
    """构造可审计的单种子短验证计划。"""
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
        "--radcil_doi_fusion_weight", "0.30",
        "--radcil_doi_align_lambda", "0.30",
        "--radcil_doi_prototype_temperature", "0.10",
    ]
    return {
        "schema_version": "stage3_wisig_hybrid_doi_seed7_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "stage": "stage3",
        "dataset": "WiSig",
        "job_id": str(job_id),
        "variant": "hybrid_radcil_doi_memory_alignment",
        "strict_protocol": (
            "Day1 60% train / 10% validation / 30% held-out evaluation; "
            "unknown-round truth is excluded from discovery, fusion and training."
        ),
        "run": {
            "experiment_entry": "experiments/exp_wisig_mvacc_cil_strict.py",
            "dataset_path": dataset_path,
            "save_dir": save_dir,
            "argv": argv,
        },
        "isolated_change": {
            "network_backend": "ratio_2p0_replay_3p0",
            "fusion_weight": 0.30,
            "prototype_alignment_lambda": 0.30,
            "prototype_temperature": 0.10,
            "description": "网络 logits 与伪标签 replay 原型概率进行 late fusion。",
        },
        "acceptance_gate": {
            "baseline_summary": "results/stage2/stage2_wisig_main_single_seed_summary_44433946.json",
            "primary": "R3 Overall Acc 不低于 seed7 MV-ACC-CIL 0.6328",
            "secondary": "R3 Old Acc 或 Forgetting 至少一项改善或接近 DOI-style 强后端参考。",
            "next_action": "通过后扩展 seed 7/13/31；未通过则保留为负消融并停止扩展。",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 WiSig hybrid DOI seed7 短验证计划。")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-path", default="数据集/WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # 保留调用方提供的路径表达；本地静态计划可使用相对路径，Slurm 运行时
    # 则会传入远端绝对根目录，避免把开发者机器路径固化到共享计划中。
    plan = build_plan(Path(args.project_root), args.dataset_path, args.save_dir, args.job_id)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"WiSig hybrid DOI seed7 计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
