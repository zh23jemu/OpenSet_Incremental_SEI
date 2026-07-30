#!/bin/bash
# Stage 12：跨天表征适配 + GPCC + 当前 RADCIL 后端的 ADS-B/LoRa seed7 完整 CIL。
#
# 仅验证 discovery-only 的收益能否传递到最终增量识别，不扩展多种子。
# 适配默认只在本脚本显式打开，且每轮只读取 Day1 known train 标签和当前
# discovery 无标签样本；held-out eval 不参与适配、聚类或伪标签决策。

#SBATCH --job-name=osei-s12-cross-day-cil
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage12/slurm_cross_day_cil_seed7_%j.out
#SBATCH --error=results/stage12/slurm_cross_day_cil_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
mkdir -p results/stage12

LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-${DATA_ROOT}/results/stage5/lora_representation_screen_seed7_44573249/supcon_rf_aug.pth}"
ADSB_ROOT="${ADSB_DATA_ROOT:-${DATA_ROOT}/数据集/ADS-B/Dataset}"
ADSB_CHECKPOINT="${ADSB_CHECKPOINT:-${DATA_ROOT}/results/stage4/adsb_main_long_radcil_seed7_44470736/closedset_adsb_90known_strict.pth}"
LORA_DIR="results/stage12/lora25_cross_day_cil_seed7_${JOB_ID}"
ADSB_DIR="results/stage12/adsb_cross_day_cil_seed7_${JOB_ID}"

"${PYTHON_BIN}" experiments/exp_wisig_mvacc_cil_strict.py \
    --dataset_profile lora25 \
    --dataset_path "${LORA_NPZ}" \
    --save_dir "${LORA_DIR}" \
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
    --cil_head_warmup_epochs 2 \
    --cil_joint_epochs 8 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0

"${PYTHON_BIN}" experiments/exp_adsb_mvacc_cil_strict.py \
    --data_root "${ADSB_ROOT}" \
    --save_dir "${ADSB_DIR}" \
    --checkpoint "${ADSB_CHECKPOINT}" \
    --initial_known_classes 90 --round_size 10 --num_rounds 3 --seed 7 \
    --backbone adsb_long --batch_size 128 --test_batch_size 256 \
    --use_supcon --supcon_weight 0.1 \
    --disable_cil_baselines --disable_visualization \
    --discovery_backend gpcc \
    --cross_day_repr_adaptation \
    --cross_day_repr_epochs 2 \
    --cross_day_repr_batch_size 128 \
    --cross_day_repr_lr 1e-5 \
    --cross_day_repr_consistency_weight 0.5 \
    --cross_day_repr_coral_weight 0.05 \
    --cil_head_warmup_epochs 2 \
    --cil_joint_epochs 8 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0

"${PYTHON_BIN}" tools/stage12_cross_day_cil_report.py \
    --lora-dir "${LORA_DIR}" \
    --adsb-dir "${ADSB_DIR}" \
    --job-id "${JOB_ID}" \
    --output "results/stage12/STAGE12_CROSS_DAY_CIL_SEED7_REPORT_${JOB_ID}.md"
