#!/bin/bash
# 阶段 6：ADS-B seed7 GPCC 聚类前端验证。
#
# 这个短任务只运行 discovery-only，不进入 Long-RADCIL 增量训练；目标是先判断
# 目标类数约束的 GPCC 是否能解决 ADS-B R2/R3 欠聚类问题。

#SBATCH --job-name=osei-s6-adsb-gpcc-disc
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage6/slurm_adsb_gpcc_discovery_seed7_%j.out
#SBATCH --error=results/stage6/slurm_adsb_gpcc_discovery_seed7_%j.err

set -euo pipefail

module load slurm
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
JOB_ID="${SLURM_JOB_ID:-manual}"
DATA_ROOT="${ADSB_DATA_ROOT:-数据集/ADS-B/Dataset}"
CHECKPOINT="results/stage4/adsb_main_long_radcil_seed7_44470736/closedset_adsb_90known_strict.pth"
MVACC_DIR="results/stage6/adsb_gpcc_discovery_mvacc_seed7_${JOB_ID}"
GPCC_DIR="results/stage6/adsb_gpcc_discovery_gpcc_seed7_${JOB_ID}"

mkdir -p results/stage6

for backend in mvacc gpcc; do
    if [[ "${backend}" == "mvacc" ]]; then
        SAVE_DIR="${MVACC_DIR}"
    else
        SAVE_DIR="${GPCC_DIR}"
    fi
    echo "运行 ADS-B discovery-only：backend=${backend}, save_dir=${SAVE_DIR}"
    "${PYTHON_BIN}" experiments/exp_adsb_mvacc_cil_strict.py \
        --data_root "${DATA_ROOT}" \
        --save_dir "${SAVE_DIR}" \
        --checkpoint "${CHECKPOINT}" \
        --initial_known_classes 90 \
        --round_size 10 \
        --num_rounds 3 \
        --seed 7 \
        --backbone adsb_long \
        --use_supcon \
        --supcon_weight 0.1 \
        --disable_cil_baselines \
        --disable_visualization \
        --discovery_only \
        --discovery_backend "${backend}"
done

"${PYTHON_BIN}" tools/stage6_gpcc_discovery_report.py \
    --dataset adsb \
    --job-id "${JOB_ID}" \
    --mvacc-save-dir "${MVACC_DIR}" \
    --gpcc-save-dir "${GPCC_DIR}" \
    --output "results/stage6/STAGE6_ADSB_GPCC_DISCOVERY_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage6/stage6_adsb_gpcc_discovery_seed7_summary_${JOB_ID}.json"
