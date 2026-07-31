#!/bin/bash
# Stage 25：LoRa recording-level 评估聚合 seed7 诊断。
#
# 本任务不改变 LoRa 训练、发现、伪标签注册和 symbol-level 正式指标；
# 只在 held-out eval 完成后，按可观测 recording_id 对同一 recording
# 的逐 symbol 预测做多数投票，判断 LoRa 低分是否被 symbol 级噪声放大。

#SBATCH --job-name=osei-s25-lora
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage25/slurm_lora_recording_eval_seed7_%j.out
#SBATCH --error=results/stage25/slurm_lora_recording_eval_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-${DATA_ROOT}/results/stage5/lora_representation_screen_seed7_44573249/supcon_rf_aug.pth}"
SAVE_DIR="results/stage25/lora_recording_eval_seed7_${JOB_ID}"

mkdir -p results/stage25 "${SAVE_DIR}"

"${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
    --dataset_profile lora25 \
    --dataset_path "${LORA_NPZ}" \
    --save_dir "${SAVE_DIR}" \
    --checkpoint "${LORA_CHECKPOINT}" \
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
    --enable_joint_discovery_refinement \
    --enable_lora_recording_eval \
    --cil_head_warmup_epochs 2 \
    --cil_joint_epochs 8 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0

"${PYTHON_BIN}" tools/stage25_lora_recording_eval_report.py \
    --save-dir "${SAVE_DIR}" \
    --job-id "${JOB_ID}" \
    --output "results/stage25/STAGE25_LORA_RECORDING_EVAL_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage25/stage25_lora_recording_eval_seed7_summary_${JOB_ID}.json"

echo "Stage 25 LoRa recording-level 评估聚合 seed7 诊断完成。"
