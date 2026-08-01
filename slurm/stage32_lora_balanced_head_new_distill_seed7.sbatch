#!/usr/bin/env bash
# Stage 32：LoRa 类均衡分类头 + 新类 logits 保持 seed7 短矩阵。
#
# Stage 31 证明单纯类均衡分类头会提高 Old/Overall，但把 New 压塌。
# 本实验只保留 epochs=1 的轻量重校准，并给当前轮新伪类样本加入重校准前
# logits 蒸馏，验证是否能保住新类边界。0 轮基线完全沿用 Stage 27 正式口径。

#SBATCH --job-name=osei-s32-lora
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=results/stage32/slurm_lora_balanced_head_distill_seed7_%j.out
#SBATCH --error=results/stage32/slurm_lora_balanced_head_distill_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python"
fi
JOB_ID="${SLURM_JOB_ID}"
mkdir -p "${PROJECT_ROOT}/results/stage32"
cd "${PROJECT_ROOT}"

run_variant() {
    local tag="$1"
    local recalibration_epochs="$2"
    local distill_weight="$3"
    local save_dir="results/stage32/lora_balanced_head_distill_${tag}_${JOB_ID}"
    mkdir -p "${save_dir}"
    "${PYTHON_BIN}" experiments/exp_wisig_mvacc_cil_strict.py \
        --dataset_profile lora25 \
        --dataset_path datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz \
        --seed 7 \
        --initial_known_classes 10 \
        --round_size 5 \
        --num_rounds 3 \
        --train_closedset --epochs 20 \
        --closedset_backbone lora_chirp \
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
        --radcil_old_new_batch_ratio 2.0 \
        --radcil_balanced_head_recalibration_epochs "${recalibration_epochs}" \
        --radcil_balanced_head_new_distill_weight "${distill_weight}" \
        --save_dir "${save_dir}"
}

run_variant "base" 0 0.0
run_variant "e1_d0p5" 1 0.5
run_variant "e1_d1p0" 1 1.0

"${PYTHON_BIN}" tools/stage32_lora_balanced_head_new_distill_report.py \
    --root results/stage32 \
    --job-id "${JOB_ID}" \
    --output "results/stage32/STAGE32_LORA_BALANCED_HEAD_NEW_DISTILL_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage32/stage32_lora_balanced_head_new_distill_seed7_summary_${JOB_ID}.json"
