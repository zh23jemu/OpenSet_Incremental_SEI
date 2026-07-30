#!/bin/bash
# Stage 18：ADS-B discovery backbone 重训 seed7 discovery-only 矩阵。
#
# 目标：不再调 RADCIL 后端，而是重训 Day1 closed-set backbone，看 GPCC 的
# discovery-only 聚类质量是否先真正超过现有静态 backbone。候选只使用 Day1
# 已知类训练/验证样本；未知轮次真实标签只用于离线 clustering metric。

#SBATCH --job-name=osei-s18-adsb-bb
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage18/slurm_adsb_backbone_seed7_%j.out
#SBATCH --error=results/stage18/slurm_adsb_backbone_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
ADSB_ROOT="${ADSB_DATA_ROOT:-${DATA_ROOT}/数据集/ADS-B/Dataset}"
ROOT_DIR="results/stage18/adsb_backbone_discovery_seed7_${JOB_ID}"
mkdir -p results/stage18 "${ROOT_DIR}"

run_candidate() {
    local name="$1"
    local supcon_weight="$2"
    local rf_aug_flag="$3"
    local save_dir="${ROOT_DIR}/${name}"
    local checkpoint="${save_dir}/closedset_adsb_90known_strict.pth"

    "${PYTHON_BIN}" experiments/exp_adsb_mvacc_cil_strict.py \
        --data_root "${ADSB_ROOT}" \
        --save_dir "${save_dir}" \
        --checkpoint "${checkpoint}" \
        --train_closedset \
        --epochs 12 \
        --initial_known_classes 90 --round_size 10 --num_rounds 3 --seed 7 \
        --backbone adsb_long --batch_size 128 --test_batch_size 256 \
        --use_supcon --supcon_weight "${supcon_weight}" \
        --discovery_backend gpcc \
        --discovery_only \
        --disable_cil_baselines --disable_visualization \
        ${rf_aug_flag}
}

run_candidate "supcon_w0p30_rfaug" "0.30" ""
run_candidate "supcon_w0p30_noaug" "0.30" "--disable_rf_augmentation"

"${PYTHON_BIN}" tools/stage18_adsb_backbone_discovery_report.py \
    --root "${ROOT_DIR}" \
    --output "results/stage18/STAGE18_ADSB_BACKBONE_DISCOVERY_SEED7_REPORT_${JOB_ID}.md"
