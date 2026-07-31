#!/bin/bash
# Stage 26：LoRa discovery 前自监督表征适配 seed7 矩阵。
#
# 该任务验证一个更大的 LoRa-specific 方向：每轮 discovery 前，在当前
# discovery 无标签样本上做 instance contrastive 自监督适配，同时用 Day1
# 已知训练标签做 CE 保底。它不读取未知真值或 held-out eval，不改变 strict split。

#SBATCH --job-name=osei-s26-lora
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage26/slurm_lora_ssl_adaptation_seed7_%j.out
#SBATCH --error=results/stage26/slurm_lora_ssl_adaptation_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-${DATA_ROOT}/results/stage5/lora_representation_screen_seed7_44573249/supcon_rf_aug.pth}"

mkdir -p results/stage26

run_variant() {
    local label="$1"
    shift
    local save_dir="results/stage26/lora_${label}_seed7_${JOB_ID}"
    mkdir -p "${save_dir}"

    "${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
        --dataset_profile lora25 \
        --dataset_path "${LORA_NPZ}" \
        --save_dir "${save_dir}" \
        --checkpoint "${LORA_CHECKPOINT}" \
        --seed 7 --initial_known_classes 10 --round_size 5 --num_rounds 3 \
        --use_supcon --supcon_weight 0.1 \
        --disable_cil_baselines --disable_visualization \
        --discovery_backend gpcc \
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
        --radcil_old_new_batch_ratio 2.0 \
        "$@"
}

run_variant "ssl_only"
run_variant "ssl_plus_cross_day" \
    --cross_day_repr_adaptation \
    --cross_day_repr_epochs 2 \
    --cross_day_repr_batch_size 128 \
    --cross_day_repr_lr 1e-5 \
    --cross_day_repr_consistency_weight 0.5 \
    --cross_day_repr_coral_weight 0.05

"${PYTHON_BIN}" tools/stage26_lora_ssl_adaptation_report.py \
    --root results/stage26 \
    --job-id "${JOB_ID}" \
    --output "results/stage26/STAGE26_LORA_SSL_ADAPTATION_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage26/stage26_lora_ssl_adaptation_seed7_summary_${JOB_ID}.json"

echo "Stage 26 LoRa SSL discovery 表征适配 seed7 矩阵完成。"
