#!/bin/bash
# 阶段 5：LoRa25 RADCIL 冻结 backbone 单因素消融。
#
# 与 Job 44573179 复用同一初始 checkpoint、strict split、MV-ACC 参数、
# replay weight 和 old:new ratio；唯一变化是 radcil_unfreeze_scope=none，
# 用于验证跨天旧类遗忘是否主要来自 backbone 漂移。

#SBATCH --job-name=osei-stage5-lora-freeze-s7
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage5-lora-freeze-s7-%j.out
#SBATCH --error=results/slurm-osei-stage5-lora-freeze-s7-%j.err

set -euo pipefail
module load slurm
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="results/stage5/lora_radcil_frozen_backbone_seed7_${JOB_ID}"
CHECKPOINT="results/stage5/lora_mvacc_cil_seed7_44573179/closedset_lora_day1_10known.pth"
LORA_NPZ="${LORA25_NPZ:-datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"

"${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
    --dataset_path "${LORA_NPZ}" \
    --save_dir "${OUTPUT_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --seed 7 \
    --initial_known_classes 10 \
    --round_size 5 \
    --num_rounds 3 \
    --use_supcon \
    --supcon_weight 0.1 \
    --disable_cil_baselines \
    --disable_visualization \
    --cil_head_warmup_epochs 2 \
    --cil_joint_epochs 8 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0 \
    --radcil_unfreeze_scope none \
    --mvacc_min_cluster_ratio 0.03 \
    --mvacc_min_samples 5

"${PYTHON_BIN}" tools/stage5_lora_mvacc_cil_report.py \
    --save-dir "${OUTPUT_DIR}" \
    --job-id "${JOB_ID}" \
    --output "results/stage5/STAGE5_LORA_RADCIL_FROZEN_BACKBONE_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage5/stage5_lora_radcil_frozen_backbone_seed7_summary_${JOB_ID}.json"
