#!/bin/bash
# Stage 36：LoRa 伪标签 prototype + 增强一致性联合训练 seed7。
#
# 固定 Stage 35 的发现和注册配置，只比较新的训练目标，避免把
# discovery、registration 和 CIL 后端同时改变导致结论不可解释。

#SBATCH --job-name=osei-s36-lora
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage36/slurm_lora_pseudo_self_training_seed7_%j.out
#SBATCH --error=results/stage36/slurm_lora_pseudo_self_training_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
CHECKPOINT="${LORA_STAGE33_CHECKPOINT:-${DATA_ROOT}/results/stage33/lora_chirp_closedset_seed7_45359726.pth}"
ROOT="results/stage36"

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}"
"${PYTHON_BIN}" tools/stage36_lora_pseudo_self_training_plan.py \
    --output "${ROOT}/stage36_lora_pseudo_self_training_plan.json"

for VARIANT in baseline proto0p10_cons0p10 proto0p20_cons0p20; do
    case "${VARIANT}" in
        baseline)
            PROTO_WEIGHT="0.0"
            CONS_WEIGHT="0.0"
            ;;
        proto0p10_cons0p10)
            PROTO_WEIGHT="0.10"
            CONS_WEIGHT="0.10"
            ;;
        proto0p20_cons0p20)
            PROTO_WEIGHT="0.20"
            CONS_WEIGHT="0.20"
            ;;
    esac
    SAVE_DIR="${ROOT}/lora_pseudo_self_training_${VARIANT}_seed7_${JOB_ID}"
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
        --radcil_new_prototype_weight "${PROTO_WEIGHT}" \
        --radcil_pseudo_aug_consistency_weight "${CONS_WEIGHT}" \
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
        --memory_per_class 20
done

"${PYTHON_BIN}" tools/stage36_lora_pseudo_self_training_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --output "${ROOT}/STAGE36_LORA_PSEUDO_SELF_TRAINING_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage36_lora_pseudo_self_training_summary_${JOB_ID}.json"

echo "Stage 36 LoRa 伪标签自训练 seed7 完成。"
