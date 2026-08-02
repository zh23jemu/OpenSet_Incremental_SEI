#!/bin/bash
# Stage 41：LoRa 原始 IQ + 幅度/相位几何双分支 backbone seed7 验证。
#
# 这是新的结构性候选：只替换 Day1 closed-set backbone，后续 strict split、
# GPCC、cross-day、LoRa SSL、联合 discovery-CIL 和 held-out 评估保持一致。

#SBATCH --job-name=osei-s41-lora
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage41/slurm_lora_hybrid_seed7_%j.out
#SBATCH --error=results/stage41/slurm_lora_hybrid_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"

mkdir -p results/stage41

run_variant() {
    local label="$1"
    local backbone="$2"
    local save_dir="results/stage41/lora_${label}_seed7_${JOB_ID}"
    local checkpoint="${save_dir}/closedset_lora25_${backbone}_10known.pth"
    mkdir -p "${save_dir}"

    "${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
        --dataset_profile lora25 \
        --dataset_path "${LORA_NPZ}" \
        --save_dir "${save_dir}" \
        --checkpoint "${checkpoint}" \
        --train_closedset --epochs 20 \
        --closedset_backbone "${backbone}" \
        --seed 7 --initial_known_classes 10 --round_size 5 --num_rounds 3 \
        --use_supcon --supcon_weight 0.1 \
        --disable_cil_baselines --disable_visualization \
        --discovery_backend gpcc \
        --cross_day_repr_adaptation \
        --cross_day_repr_epochs 2 \
        --cross_day_repr_batch_size 128 \
        --cross_day_repr_lr 1e-5 \
        --cross_day_repr_consistency_weight 0.5 \
        --cross_day_repr_coral_weight 0.05 \
        --lora_ssl_adaptation \
        --lora_ssl_epochs 3 \
        --lora_ssl_batch_size 128 \
        --lora_ssl_lr 1e-5 \
        --lora_ssl_ce_weight 1.0 \
        --lora_ssl_instance_weight 0.5 \
        --lora_ssl_teacher_weight 0.2 \
        --lora_ssl_temperature 0.2 \
        --enable_joint_discovery_refinement \
        --enable_lora_recording_eval \
        --cil_head_warmup_epochs 2 \
        --cil_joint_epochs 8 \
        --cil_replay_weight 3.0 \
        --radcil_old_new_batch_ratio 2.0
}

run_variant "chirp" "lora_chirp"
run_variant "hybrid" "lora_hybrid"

"${PYTHON_BIN}" tools/stage41_lora_hybrid_backbone_report.py \
    --root results/stage41 \
    --job-id "${JOB_ID}" \
    --output "results/stage41/STAGE41_LORA_HYBRID_BACKBONE_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage41/stage41_lora_hybrid_backbone_seed7_summary_${JOB_ID}.json"

echo "Stage 41 LoRa hybrid backbone seed7 完成。"
