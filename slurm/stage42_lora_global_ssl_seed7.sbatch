#!/bin/bash
# Stage 42：LoRa 跨天 discovery 全局实例对比预训练 seed7。
# 只使用 Day1 IQ_1-6 与 Day2-4 IQ_1-7 discovery，不读取 held-out eval。

#SBATCH --job-name=osei-s42-lora
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage42/slurm_lora_global_ssl_seed7_%j.out
#SBATCH --error=results/stage42/slurm_lora_global_ssl_seed7_%j.err

set -euo pipefail
module load slurm
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
ROOT="results/stage42"
CHECKPOINT="${ROOT}/lora_global_ssl_checkpoint_seed7_${JOB_ID}.pth"

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}"
"${PYTHON_BIN}" tools/stage42_lora_global_ssl_pretrain.py \
  --dataset-path "${LORA_NPZ}" \
  --output "${CHECKPOINT}" \
  --summary-json "${ROOT}/stage42_lora_global_ssl_checkpoint_summary_${JOB_ID}.json" \
  --seed 7 --pretrain-epochs 12 --finetune-epochs 12 --batch-size 128

SAVE_DIR="${ROOT}/lora_global_ssl_seed7_${JOB_ID}"
mkdir -p "${SAVE_DIR}"
"${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
  --dataset_profile lora25 --dataset_path "${LORA_NPZ}" \
  --save_dir "${SAVE_DIR}" --checkpoint "${CHECKPOINT}" --closedset_backbone lora_chirp \
  --seed 7 --initial_known_classes 10 --round_size 5 --num_rounds 3 \
  --use_supcon --supcon_weight 0.1 --discovery_backend gpcc \
  --cross_day_repr_adaptation --cross_day_repr_epochs 2 --cross_day_repr_batch_size 128 \
  --cross_day_repr_lr 1e-5 --cross_day_repr_consistency_weight 0.5 --cross_day_repr_coral_weight 0.05 \
  --lora_ssl_adaptation --lora_ssl_epochs 3 --lora_ssl_batch_size 128 --lora_ssl_lr 1e-5 \
  --lora_ssl_ce_weight 1.0 --lora_ssl_instance_weight 0.5 --lora_ssl_teacher_weight 0.2 \
  --lora_ssl_temperature 0.2 --enable_joint_discovery_refinement --enable_lora_recording_eval \
  --cil_head_warmup_epochs 2 --cil_joint_epochs 8 --cil_replay_weight 3.0 \
  --radcil_old_new_batch_ratio 2.0

"${PYTHON_BIN}" tools/stage42_lora_global_ssl_report.py \
  --root "${ROOT}" --job-id "${JOB_ID}" \
  --output "${ROOT}/STAGE42_LORA_GLOBAL_SSL_SEED7_REPORT_${JOB_ID}.md" \
  --summary-json "${ROOT}/stage42_lora_global_ssl_summary_${JOB_ID}.json"
