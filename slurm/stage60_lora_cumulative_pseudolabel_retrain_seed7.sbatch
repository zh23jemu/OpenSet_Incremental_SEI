#!/bin/bash
# Stage 60：LoRa 累积伪标签全量重训 seed7 验证。
#
# 背景：
#   Stage57/58 说明后验路由仍是 Old/New 跷跷板，Stage59 说明传统 RF
#   物理统计原型链路远弱于 Chirp backbone。本阶段绕开 RADCIL 增量后端：
#   每轮先用上一轮 LoRaChirp 表征做固定 K=5 discovery，再把所有已见
#   真/伪标签样本合并，重新训练一个 10/15/20/25 类分类器。
#
# 通过门槛：
#   相对 Stage48 三种子，R3 Overall 至少 +0.01；Old 不明显下降；
#   New 下降不超过 0.05；Forgetting 恶化不超过 0.05。否则不扩三种子。
#SBATCH --job-name=stage60_lora_cum_retrain
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage60/slurm_lora_cumulative_pseudolabel_retrain_seed7_%j.out
#SBATCH --error=results/stage60/slurm_lora_cumulative_pseudolabel_retrain_seed7_%j.err

set -euo pipefail

module load slurm

DATA_ROOT="${DATA_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python"
fi
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 127
fi

JOB_ID="${SLURM_JOB_ID:-manual}"
ROOT="results/stage60"
S28_NPZ="${LORA25_S28_NPZ:-results/stage46/lora25_diffdays_indoor_raw1024_s28_46103936.npz}"
SAVE_DIR="${ROOT}/lora_s60_cumulative_seed7_${JOB_ID}"
mkdir -p "${SAVE_DIR}"

echo "Stage60 data_root=${DATA_ROOT}"
echo "Stage60 s28_npz=${S28_NPZ}"
echo "Stage60 python=${PYTHON_BIN}"
echo "Stage60 save_dir=${SAVE_DIR}"
if [ ! -f "${S28_NPZ}" ]; then
    echo "Missing s28 NPZ: ${S28_NPZ}. Run Stage46 first or pass LORA25_S28_NPZ." >&2
    exit 2
fi

"${PYTHON_BIN}" tools/stage60_lora_cumulative_pseudolabel_retrain.py \
    --dataset-path "${S28_NPZ}" \
    --save-dir "${SAVE_DIR}" \
    --seed 7 \
    --epochs 10 \
    --batch-size 96 \
    --test-batch-size 192 \
    --lr 1e-3 \
    --recording-consensus-threshold 0.65 \
    --use-supcon \
    --supcon-weight 0.1
