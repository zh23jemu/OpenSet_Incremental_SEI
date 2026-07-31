#!/bin/bash
# Stage 27：LoRa-specific chirp backbone seed7 对比。
#
# 只替换 Day1 closed-set backbone，后续 GPCC、cross-day、联合 discovery-CIL
# 和评估口径保持一致。resnet1d 是同任务内重训对照，chirp 是新结构候选。

#SBATCH --job-name=osei-s27-lora
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage27/slurm_lora_backbone_seed7_%j.out
#SBATCH --error=results/stage27/slurm_lora_backbone_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"

mkdir -p results/stage27

run_variant() {
    local label="$1"
    local backbone="$2"
    local save_dir="results/stage27/lora_${label}_seed7_${JOB_ID}"
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

run_variant "resnet1d" "resnet1d"
run_variant "chirp" "lora_chirp"

"${PYTHON_BIN}" tools/stage27_lora_backbone_report.py \
    --root results/stage27 \
    --job-id "${JOB_ID}" \
    --output "results/stage27/STAGE27_LORA_BACKBONE_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage27/stage27_lora_backbone_seed7_summary_${JOB_ID}.json"

echo "Stage 27 LoRa backbone seed7 对比完成。"
