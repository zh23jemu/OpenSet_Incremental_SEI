#!/bin/bash
# 阶段 6：LoRa seed7 GPCC 聚类前端验证。
#
# 这个短任务只运行 discovery-only，不进入增量训练后端；目标是先回答
# “换掉 HDBSCAN 以后，聚类本身有没有变好”。同一个作业内顺序跑旧 MV-ACC
# 与新 GPCC，方便报告器做同环境对比。

#SBATCH --job-name=osei-s6-lora-gpcc-disc
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=results/stage6/slurm_lora_gpcc_discovery_seed7_%j.out
#SBATCH --error=results/stage6/slurm_lora_gpcc_discovery_seed7_%j.err

set -euo pipefail

module load slurm
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
CHECKPOINT="${LORA_GPCC_CHECKPOINT:-results/stage5/lora_representation_screen_seed7_44573249/supcon_rf_aug.pth}"
MVACC_DIR="results/stage6/lora_gpcc_discovery_mvacc_seed7_${JOB_ID}"
GPCC_DIR="results/stage6/lora_gpcc_discovery_gpcc_seed7_${JOB_ID}"

mkdir -p results/stage6

for backend in mvacc gpcc; do
    if [[ "${backend}" == "mvacc" ]]; then
        SAVE_DIR="${MVACC_DIR}"
    else
        SAVE_DIR="${GPCC_DIR}"
    fi
    echo "运行 LoRa discovery-only：backend=${backend}, save_dir=${SAVE_DIR}"
    "${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
        --dataset_path "${LORA_NPZ}" \
        --save_dir "${SAVE_DIR}" \
        --checkpoint "${CHECKPOINT}" \
        --seed 7 \
        --initial_known_classes 10 \
        --round_size 5 \
        --num_rounds 3 \
        --use_supcon \
        --supcon_weight 0.1 \
        --disable_cil_baselines \
        --disable_visualization \
        --discovery_only \
        --discovery_backend "${backend}"
done

"${PYTHON_BIN}" tools/stage6_gpcc_discovery_report.py \
    --dataset lora \
    --job-id "${JOB_ID}" \
    --mvacc-save-dir "${MVACC_DIR}" \
    --gpcc-save-dir "${GPCC_DIR}" \
    --output "results/stage6/STAGE6_LORA_GPCC_DISCOVERY_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage6/stage6_lora_gpcc_discovery_seed7_summary_${JOB_ID}.json"
