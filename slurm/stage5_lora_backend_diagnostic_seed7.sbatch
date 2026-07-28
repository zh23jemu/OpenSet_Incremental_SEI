#!/bin/bash
# 阶段 5：LoRa25 seed7 严格后端诊断。
#
# 复用正式 Job 44573179 的初始闭集 checkpoint，在相同 strict split、
# 相同 MV-ACC 伪标签与冻结初始表征下运行代表性 CIL 后端。该作业用于
# 分离“跨天表征上限”和“RADCIL 遗忘”两类风险，不使用评估真值选参。

#SBATCH --job-name=osei-stage5-lora-backend-s7
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage5-lora-backend-s7-%j.out
#SBATCH --error=results/slurm-osei-stage5-lora-backend-s7-%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="results/stage5/lora_backend_diagnostic_seed7_${JOB_ID}"
CHECKPOINT="results/stage5/lora_mvacc_cil_seed7_44573179/closedset_lora_day1_10known.pth"
LORA_NPZ="${LORA25_NPZ:-datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"

mkdir -p "${OUTPUT_DIR}"
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "缺少正式 seed7 checkpoint：${CHECKPOINT}" >&2
    exit 2
fi

"${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
    --dataset_path "${LORA_NPZ}" \
    --save_dir "${OUTPUT_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --seed 7 \
    --initial_known_classes 10 \
    --round_size 5 \
    --num_rounds 3 \
    --use_supcon \
    --supcon_weight 0.1 \
    --disable_visualization \
    --cil_epochs 12 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0 \
    --mvacc_min_cluster_ratio 0.03 \
    --mvacc_min_samples 5

echo "LoRa25 seed7 后端诊断完成：${PROJECT_ROOT}/${OUTPUT_DIR}"
