#!/bin/bash
# Stage 11：训练期跨天表征适配 discovery-only seed7。
#
# 每个数据集只比较 none/cross_day 两种方法，使用 GPCC 完成固定簇数聚类。
# 该任务不跑 CIL，先验证跨天表征适配是否真的改善 discovery，再决定是否进入后端。

#SBATCH --job-name=osei-s11-cross-day
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage11/slurm_cross_day_repr_seed7_%j.out
#SBATCH --error=results/stage11/slurm_cross_day_repr_seed7_%j.err

set -euo pipefail
module load slurm

# 使用独立 worktree 执行脚本，但数据和 checkpoint 仍从主项目绝对路径读取，
# 避免 worktree 没有 .venv 或大文件时出现隐式回退到旧代码的问题。
# 默认使用提交作业时所在的 worktree；如果从外部封装脚本提交，
# 仍可通过 PROJECT_ROOT 显式覆盖，避免路径写死到旧 worktree。
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
mkdir -p results/stage11

LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-${DATA_ROOT}/results/stage5/lora_representation_screen_seed7_44573249/supcon_rf_aug.pth}"
ADSB_ROOT="${ADSB_DATA_ROOT:-${DATA_ROOT}/数据集/ADS-B/Dataset}"
ADSB_CHECKPOINT="${ADSB_CHECKPOINT:-${DATA_ROOT}/results/stage4/adsb_main_long_radcil_seed7_44470736/closedset_adsb_90known_strict.pth}"

for method in none cross_day; do
    LORA_DIR="results/stage11/lora25_${method}_seed7_${JOB_ID}"
    ADSB_DIR="results/stage11/adsb_${method}_seed7_${JOB_ID}"

    "${PYTHON_BIN}" tools/stage11_cross_day_repr_discovery.py \
        --dataset_profile lora25 \
        --dataset_path "${LORA_NPZ}" \
        --checkpoint "${LORA_CHECKPOINT}" \
        --save_dir "${LORA_DIR}" \
        --method "${method}" \
        --seed 7 --feat_dim 128 --epochs 2 --batch_size 128 --test_batch_size 256 \
        --lr 1e-5 --consistency_weight 0.5 --coral_weight 0.05

    "${PYTHON_BIN}" tools/stage11_cross_day_repr_discovery.py \
        --dataset_profile adsb \
        --data_root "${ADSB_ROOT}" \
        --checkpoint "${ADSB_CHECKPOINT}" \
        --save_dir "${ADSB_DIR}" \
        --method "${method}" \
        --seed 7 --feat_dim 128 --epochs 2 --batch_size 128 --test_batch_size 256 \
        --lr 1e-5 --consistency_weight 0.5 --coral_weight 0.05
done

"${PYTHON_BIN}" tools/stage11_cross_day_repr_report.py \
    --root results/stage11 \
    --job-id "${JOB_ID}" \
    --output "results/stage11/STAGE11_CROSS_DAY_REPR_SEED7_REPORT_${JOB_ID}.md"
