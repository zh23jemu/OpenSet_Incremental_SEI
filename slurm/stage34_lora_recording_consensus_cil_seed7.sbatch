#!/bin/bash
# Stage 34：LoRa recording-consensus=0.65 完整 seed7 CIL。
#
# Stage 33 discovery-only 已通过门槛，本任务只验证聚类收益能否传到
# Overall/Old/New。使用 Stage 33 共享的 closed-set Chirp checkpoint，
# 不重复训练初始 backbone，也不改变 strict 数据边界。

#SBATCH --job-name=osei-s34-lora
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage34/slurm_lora_recording_consensus_seed7_%j.out
#SBATCH --error=results/stage34/slurm_lora_recording_consensus_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
CHECKPOINT="${LORA_STAGE33_CHECKPOINT:-${DATA_ROOT}/results/stage33/lora_chirp_closedset_seed7_45359726.pth}"
SAVE_DIR="results/stage34/lora_recording_consensus_cil_seed7_${JOB_ID}"

cd "${PROJECT_ROOT}"
mkdir -p results/stage34 "${SAVE_DIR}"

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

"${PYTHON_BIN}" tools/stage34_lora_recording_consensus_cil_report.py \
    --save-dir "${SAVE_DIR}" \
    --job-id "${JOB_ID}" \
    --output "results/stage34/STAGE34_LORA_RECORDING_CONSENSUS_CIL_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage34/stage34_lora_recording_consensus_cil_seed7_summary_${JOB_ID}.json"

echo "Stage 34 LoRa recording-consensus seed7 完整 CIL 完成。"
