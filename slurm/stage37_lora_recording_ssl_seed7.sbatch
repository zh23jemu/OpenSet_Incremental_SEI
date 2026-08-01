#!/bin/bash
# Stage 37：LoRa recording-level SSL seed7 矩阵。
#
# 本阶段把改动放在 discovery 前的表征适配：同一 recording_id 的多个
# symbol 互为正样本，未知真值和 held-out eval 均不进入训练。

#SBATCH --job-name=osei-s37-lora
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage37/slurm_lora_recording_ssl_seed7_%j.out
#SBATCH --error=results/stage37/slurm_lora_recording_ssl_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
CHECKPOINT="${LORA_STAGE33_CHECKPOINT:-${DATA_ROOT}/results/stage33/lora_chirp_closedset_seed7_45359726.pth}"
ROOT="results/stage37"

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}"
"${PYTHON_BIN}" tools/stage37_lora_recording_ssl_plan.py \
    --output "${ROOT}/stage37_lora_recording_ssl_plan.json"

for VARIANT in baseline recording_ssl_rec0p50 recording_ssl_mix0p50; do
    EXTRA_ARGS=()
    case "${VARIANT}" in
        baseline)
            ;;
        recording_ssl_rec0p50)
            EXTRA_ARGS+=(
                --lora_recording_ssl_adaptation
                --lora_recording_ssl_recording_weight 0.50
                --lora_recording_ssl_instance_weight 0.0
            )
            ;;
        recording_ssl_mix0p50)
            EXTRA_ARGS+=(
                --lora_recording_ssl_adaptation
                --lora_recording_ssl_recording_weight 0.50
                --lora_recording_ssl_instance_weight 0.20
            )
            ;;
    esac
    SAVE_DIR="${ROOT}/lora_recording_ssl_${VARIANT}_seed7_${JOB_ID}"
    mkdir -p "${SAVE_DIR}"
    "${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
        --dataset_profile lora25 \
        --dataset_path "${LORA_NPZ}" \
        --save_dir "${SAVE_DIR}" \
        --checkpoint "${CHECKPOINT}" \
        --closedset_backbone lora_chirp \
        --seed 7 --initial_known_classes 10 --round_size 5 --num_rounds 3 \
        --use_supcon --supcon_weight 0.1 \
        --discovery_backend gpcc_recording_consensus \
        --recording_consensus_threshold 0.65 \
        --radcil_registration_top_fraction 0.80 \
        --radcil_registration_min_per_class 1 \
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
        "${EXTRA_ARGS[@]}" \
        --enable_joint_discovery_refinement \
        --enable_lora_recording_eval \
        --cil_head_warmup_epochs 2 \
        --cil_joint_epochs 8 \
        --cil_replay_weight 3.0 \
        --radcil_old_new_batch_ratio 2.0 \
        --memory_per_class 20
done

"${PYTHON_BIN}" tools/stage37_lora_recording_ssl_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --output "${ROOT}/STAGE37_LORA_RECORDING_SSL_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage37_lora_recording_ssl_summary_${JOB_ID}.json"

echo "Stage 37 LoRa recording-level SSL seed7 完成。"
