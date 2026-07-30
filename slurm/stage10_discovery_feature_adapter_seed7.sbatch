#!/bin/bash
# Stage 10：ADS-B / LoRa discovery 特征适配器 seed7 短矩阵。
#
# 只跑 GPCC discovery-only，不进入 CIL。每个数据集比较 none、
# mn_smooth、proto_repulse 三种前端，避免把低质量伪标签带入后端。

#SBATCH --job-name=osei-s10-discovery-adapter
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage10/slurm_discovery_feature_adapter_seed7_%j.out
#SBATCH --error=results/stage10/slurm_discovery_feature_adapter_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
mkdir -p results/stage10

LORA_NPZ="${LORA25_NPZ:-datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-results/stage5/lora_representation_screen_seed7_44573249/supcon_rf_aug.pth}"
ADSB_ROOT="${ADSB_DATA_ROOT:-数据集/ADS-B/Dataset}"
ADSB_CHECKPOINT="${ADSB_CHECKPOINT:-results/stage4/adsb_main_long_radcil_seed7_44470736/closedset_adsb_90known_strict.pth}"

for adapter in none mn_smooth proto_repulse; do
    LORA_DIR="results/stage10/lora25_${adapter}_seed7_${JOB_ID}"
    ADSB_DIR="results/stage10/adsb_${adapter}_seed7_${JOB_ID}"

    "${PYTHON_BIN}" experiments/exp_wisig_mvacc_cil_strict.py \
        --dataset_profile lora25 \
        --dataset_path "${LORA_NPZ}" \
        --save_dir "${LORA_DIR}" \
        --checkpoint "${LORA_CHECKPOINT}" \
        --seed 7 --initial_known_classes 10 --round_size 5 --num_rounds 3 \
        --use_supcon --supcon_weight 0.1 \
        --disable_cil_baselines --disable_visualization --discovery_only \
        --discovery_backend gpcc \
        --discovery_feature_adapter "${adapter}" \
        --discovery_adapter_smooth_k 12 \
        --discovery_adapter_smooth_weight 0.20 \
        --discovery_adapter_repulsion_weight 0.15

    "${PYTHON_BIN}" experiments/exp_adsb_mvacc_cil_strict.py \
        --data_root "${ADSB_ROOT}" \
        --save_dir "${ADSB_DIR}" \
        --checkpoint "${ADSB_CHECKPOINT}" \
        --initial_known_classes 90 --round_size 10 --num_rounds 3 --seed 7 \
        --backbone adsb_long --batch_size 128 --test_batch_size 256 \
        --use_supcon --supcon_weight 0.1 \
        --disable_cil_baselines --disable_visualization --discovery_only \
        --discovery_backend gpcc \
        --discovery_feature_adapter "${adapter}" \
        --discovery_adapter_smooth_k 12 \
        --discovery_adapter_smooth_weight 0.20 \
        --discovery_adapter_repulsion_weight 0.15
done

"${PYTHON_BIN}" tools/stage10_discovery_feature_adapter_report.py \
    --root results/stage10 --job-id "${JOB_ID}" \
    --output "results/stage10/STAGE10_DISCOVERY_FEATURE_ADAPTER_SEED7_REPORT_${JOB_ID}.md"
