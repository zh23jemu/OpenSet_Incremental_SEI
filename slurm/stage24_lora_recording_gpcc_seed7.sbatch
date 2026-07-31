#!/bin/bash
# Stage 24：LoRa recording-level GPCC seed7 结构性验证。
#
# 每次 LoRa IQ transmission 会生成多个 aligned symbol。该任务只使用
# loader 中可观测的 recording_id，把同一次录制的 symbol 特征聚合后再跑
# 固定 K=5 的 GPCC，随后回填样本级伪标签进入原 RADCIL 链路。
#
# 不读取当前轮未知真值或 IQ_8-10 held-out evaluation；真实标签仅用于
# 训练结束后的离线聚类/识别指标。候选若未同时守住 Overall、Old、New，
# 不再扩展三种子。

#SBATCH --job-name=osei-s24-lora-recording
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage24/slurm_lora_recording_gpcc_seed7_%j.out
#SBATCH --error=results/stage24/slurm_lora_recording_gpcc_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-${DATA_ROOT}/results/stage5/lora_representation_screen_seed7_44573249/supcon_rf_aug.pth}"
SAVE_DIR="results/stage24/lora_recording_gpcc_seed7_${JOB_ID}"

cd "${PROJECT_ROOT}"
mkdir -p results/stage24 "${SAVE_DIR}"

"${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
    --dataset_profile lora25 \
    --dataset_path "${LORA_NPZ}" \
    --save_dir "${SAVE_DIR}" \
    --checkpoint "${LORA_CHECKPOINT}" \
    --seed 7 --initial_known_classes 10 --round_size 5 --num_rounds 3 \
    --use_supcon --supcon_weight 0.1 \
    --disable_cil_baselines --disable_visualization \
    --discovery_backend gpcc_recording \
    --cross_day_repr_adaptation \
    --cross_day_repr_epochs 2 \
    --cross_day_repr_batch_size 128 \
    --cross_day_repr_lr 1e-5 \
    --cross_day_repr_consistency_weight 0.5 \
    --cross_day_repr_coral_weight 0.05 \
    --enable_joint_discovery_refinement \
    --cil_head_warmup_epochs 2 \
    --cil_joint_epochs 8 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0

"${PYTHON_BIN}" tools/stage24_lora_recording_gpcc_report.py \
    --save-dir "${SAVE_DIR}" \
    --job-id "${JOB_ID}" \
    --output "results/stage24/STAGE24_LORA_RECORDING_GPCC_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage24/stage24_lora_recording_gpcc_seed7_summary_${JOB_ID}.json"

echo "Stage 24 LoRa recording-level GPCC seed7 完成。"
