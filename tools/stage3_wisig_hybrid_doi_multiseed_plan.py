"""生成 hybrid RADCIL + DOI-memory 正式三种子计划。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def experiment_args(dataset_path: str, save_dir: str, seed: int) -> list[str]:
    """返回与通过门槛的 Job 44465809 完全一致的实验参数。"""
    return [
        "--dataset_path", dataset_path,
        "--save_dir", save_dir,
        "--selected_rx_list", "2",
        "--initial_known_classes", "10",
        "--round_size", "10",
        "--num_rounds", "3",
        "--seed", str(seed),
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


def build_plan(project_root: Path, dataset_path: str, output_prefix: str, job_id: str) -> dict:
    """构造 seed 7/13/31 固定配置计划。"""
    seeds = (7, 13, 31)
    runs = []
    for seed in seeds:
        save_dir = f"{output_prefix}_seed{seed}_{job_id}"
        runs.append(
            {
                "run_name": f"wisig_hybrid_doi_seed{seed}",
                "seed": seed,
                "variant": "hybrid_radcil_doi_memory_alignment",
                "frontend": "MV-ACC",
                "save_dir": save_dir,
                "argv": experiment_args(dataset_path, save_dir, seed),
            }
        )
    return {
        "schema_version": "stage3_wisig_hybrid_doi_multiseed_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "stage": "stage3",
        "dataset": "WiSig",
        "task": "formal 3-seed hybrid RADCIL + DOI-memory validation",
        "seeds": list(seeds),
        "selection_basis": {
            "variant_role": "formal_hybrid_doi",
            "source_job": "44465809",
            "decision": (
                "seed7 R3 Overall 提升 0.0072、New 提升 0.0756、Forgetting 降低 0.0022，"
                "达到预设门槛，固定参数扩展三种子。"
            ),
        },
        "locked_config": {
            "variant": "hybrid_radcil_doi_memory_alignment",
            "frontend": "MV-ACC",
            "network_backend": "ratio_2p0_replay_3p0",
            "fusion_weight": 0.30,
            "prototype_alignment_lambda": 0.30,
            "prototype_temperature": 0.10,
        },
        "runs": runs,
        "risk_controls": [
            "参数与通过 seed7 门槛的 Job 44465809 完全一致。",
            "不根据 seed 13/31 中间结果调参。",
            "报告必须包含三种子明细、均值和标准差。",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 hybrid DOI 正式三种子计划。")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-path", default="数据集/WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl")
    parser.add_argument("--output-prefix", default="results/stage3/wisig_hybrid_doi_multiseed")
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_plan(Path(args.project_root), args.dataset_path, args.output_prefix, args.job_id)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Hybrid DOI 正式三种子计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
