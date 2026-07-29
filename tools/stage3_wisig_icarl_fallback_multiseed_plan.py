"""生成 WiSig RADCIL + iCaRL fallback 三种子正式验证计划。

该计划只在阶段 3 主方法 `ratio_2p0_replay_3p0` 基础上增加
confidence-gated exemplar fallback，不改变 MV-ACC 发现前端、训练轮数、
replay 强度、old:new batch ratio 或严格评估协议。seed7 已达到预注册
门槛后，才进入本脚本定义的 seed 7/13/31 正式验证。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DATASET = "数据集/WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl"
DEFAULT_SEEDS = (7, 13, 31)


def parse_seeds(text: str) -> tuple[int, ...]:
    """解析逗号分隔 seed，并保持输入顺序去重。"""
    seeds: list[int] = []
    for item in text.split(","):
        value = item.strip()
        if not value:
            continue
        seed = int(value)
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise ValueError("至少需要一个 seed。")
    return tuple(seeds)


def experiment_argv(dataset_path: str, save_dir: str, seed: int) -> list[str]:
    """构造单个 seed 的实验参数，确保和阶段 3 主后端只差 fallback 开关。"""
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


def build_plan(
    project_root: Path,
    dataset_path: str,
    output_prefix: str,
    job_id: str,
    seeds: tuple[int, ...],
) -> dict:
    """构造可审计的三种子验证计划。"""
    runs = []
    for seed in seeds:
        save_dir = f"{output_prefix}_seed{seed}_{job_id}"
        runs.append({
            "run_name": f"wisig_icarl_fallback_seed{seed}",
            "seed": int(seed),
            "save_dir": save_dir,
            "argv": experiment_argv(dataset_path, save_dir, seed),
        })
    return {
        "schema_version": "stage3_wisig_icarl_fallback_multiseed_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "stage": "stage3",
        "dataset": "WiSig",
        "variant": "hybrid_radcil_icarl_exemplar_classifier_fallback",
        "seeds": list(seeds),
        "selection_basis": {
            "seed7_job": "44771723",
            "seed7_report": "results/stage3/STAGE3_WISIG_ICARL_FALLBACK_SEED7_REPORT_44771723.md",
            "decision": (
                "seed7 R3 Overall、Old、Forgetting 和 Macro F1 均达到预注册扩展门槛；"
                "因此扩展到 seed 7/13/31，检验收益是否跨种子稳定。"
            ),
        },
        "strict_protocol": (
            "Day1 60% train / 10% validation / 30% held-out evaluation；"
            "fallback 阈值只用 Day1 validation 选择，held-out 真值只用于最终审计。"
        ),
        "isolated_change": {
            "baseline_backend": "ratio_2p0_replay_3p0",
            "fallback": "confidence-gated replay exemplar prototype classifier",
            "threshold_candidates": [0.45, 0.55, 0.65, 0.75, 0.85],
            "tie_break": "Validation Accuracy 并列时选择更低阈值，减少 fallback 触发。",
        },
        "runs": runs,
        "acceptance_gate": {
            "baseline_summary": "results/stage3/stage3_wisig_main_multiseed_summary_44440345.json",
            "primary": "R3 Overall mean 不低于阶段 3 主方法三种子均值。",
            "secondary": "R3 Old mean 提升或 Forgetting mean 降低；若只提升 New Acc，则不作为主后端。",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 WiSig iCaRL fallback 三种子正式验证计划。")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET)
    parser.add_argument("--output-prefix", default="results/stage3/wisig_icarl_fallback_multiseed")
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--seeds", default="7,13,31")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plan = build_plan(
        project_root=project_root,
        dataset_path=args.dataset_path,
        output_prefix=args.output_prefix,
        job_id=args.job_id,
        seeds=parse_seeds(args.seeds),
    )
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"WiSig iCaRL fallback 三种子计划：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
