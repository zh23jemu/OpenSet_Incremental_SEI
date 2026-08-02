#!/bin/bash
# Stage 40：LoRa 域对抗表征预训练 seed7。
#
# Day2-4 discovery 样本只提供 Day 域标签，域头通过梯度反转训练；
# 不使用未知设备类别标签，不读取任何 IQ_8-10 held-out evaluation。

#SBATCH --job-name=osei-s40-lora
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage40/slurm_lora_domain_adversarial_seed7_%j.out
#SBATCH --error=results/stage40/slurm_lora_domain_adversarial_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
ROOT="results/stage40"
CHECKPOINT="${ROOT}/lora_domain_adversarial_checkpoint_seed7_${JOB_ID}.pth"
SAVE_DIR="${ROOT}/lora_domain_adversarial_seed7_${JOB_ID}"

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}" "${SAVE_DIR}"

"${PYTHON_BIN}" tools/stage38_lora_physical_pretrain.py \
    --dataset-path "${LORA_NPZ}" \
    --output "${CHECKPOINT}" \
    --summary-json "${ROOT}/stage40_lora_domain_adversarial_checkpoint_summary_${JOB_ID}.json" \
    --seed 7 \
    --epochs 20 \
    --batch-size 128 \
    --lr 1e-3 \
    --supcon-weight 0.1 \
    --supcon-temperature 0.2 \
    --physical-weight 0.10 \
    --domain-adversarial \
    --domain-adversarial-weight 0.10 \
    --domain-adversarial-lambda 1.0

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
    --enable_joint_discovery_refinement \
    --enable_lora_recording_eval \
    --cil_head_warmup_epochs 2 \
    --cil_joint_epochs 8 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0 \
    --memory_per_class 20

"${PYTHON_BIN}" tools/stage40_lora_domain_adversarial_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --output "${ROOT}/STAGE40_LORA_DOMAIN_ADVERSARIAL_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage40_lora_domain_adversarial_seed7_summary_${JOB_ID}.json"

echo "Stage 40 LoRa 域对抗表征预训练 seed7 完成。"
