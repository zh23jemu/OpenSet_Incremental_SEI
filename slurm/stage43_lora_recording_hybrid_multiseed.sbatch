#!/bin/bash
# Stage 43：LoRa recording-level hybrid 三种子验证。
#
# Stage 41 的 seed7 显示 hybrid 在 symbol-level 下降，但 recording-level
# Overall 有局部提升。本任务只验证这个 recording-level 口径是否跨 seed
# 稳定；不改变 strict split，不读取 held-out eval 真值调参。

#SBATCH --job-name=osei-s43-lora-rec
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage43/slurm_lora_recording_hybrid_%j.out
#SBATCH --error=results/stage43/slurm_lora_recording_hybrid_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
ROOT="results/stage43"

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}"

for seed in 7 13 31; do
  SAVE_DIR="${ROOT}/lora_hybrid_recording_seed${seed}_${JOB_ID}"
  CHECKPOINT="${SAVE_DIR}/closedset_lora25_lora_hybrid_10known.pth"
  mkdir -p "${SAVE_DIR}"
  echo "运行 Stage 43 LoRa hybrid recording：seed=${seed}"
  "${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
    --dataset_profile lora25 \
    --dataset_path "${LORA_NPZ}" \
    --save_dir "${SAVE_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --train_closedset --epochs 20 \
    --closedset_backbone lora_hybrid \
    --seed "${seed}" --initial_known_classes 10 --round_size 5 --num_rounds 3 \
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
done

"${PYTHON_BIN}" tools/stage43_lora_recording_hybrid_multiseed_report.py \
  --root "${ROOT}" \
  --job-id "${JOB_ID}" \
  --output "${ROOT}/STAGE43_LORA_RECORDING_HYBRID_MULTISEED_REPORT_${JOB_ID}.md" \
  --summary-json "${ROOT}/stage43_lora_recording_hybrid_multiseed_summary_${JOB_ID}.json"
