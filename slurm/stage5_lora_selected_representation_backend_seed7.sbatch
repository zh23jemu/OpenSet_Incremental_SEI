#!/bin/bash
# LoRa25：IQ_7 预选表征上的共享发现后端诊断。
# checkpoint 与候选配方已由 Job 44573249 只使用 IQ_7 确定；本作业
# 比较 DOI-style/iCaRL 等稳定后端，IQ_8-10 只用于最终报告。
#SBATCH --job-name=osei-stage5-lora-selected-backend-s7
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=results/slurm-osei-stage5-lora-selected-backend-s7-%j.out
#SBATCH --error=results/slurm-osei-stage5-lora-selected-backend-s7-%j.err

set -euo pipefail
module load slurm
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="results/stage5/lora_selected_representation_backend_seed7_${JOB_ID}"
CHECKPOINT="results/stage5/lora_representation_screen_seed7_44573249/supcon_rf_aug.pth"

"${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
    --dataset_path datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz \
    --save_dir "${OUTPUT_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --seed 7 \
    --initial_known_classes 10 \
    --round_size 5 \
    --num_rounds 3 \
    --use_supcon \
    --supcon_weight 0.1 \
    --disable_visualization \
    --cil_epochs 12 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0 \
    --mvacc_min_cluster_ratio 0.03 \
    --mvacc_min_samples 5
