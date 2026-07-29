#!/bin/bash
# 阶段 6：ADS-B GPCC + Long-RADCIL seed7 完整增量验证。
#
# 前置条件：`stage6_adsb_gpcc_discovery_seed7.sbatch` 已证明 GPCC 在 seed7
# discovery-only 中通过聚类门槛。该任务继续使用同一个 seed7 closed-set
# checkpoint，只把发现前端切到 GPCC，后端保持当前 ADS-B Long-RADCIL 参数。

#SBATCH --job-name=osei-s6-adsb-gpcc-cil
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage6/slurm_adsb_gpcc_cil_seed7_%j.out
#SBATCH --error=results/stage6/slurm_adsb_gpcc_cil_seed7_%j.err

set -euo pipefail

module load slurm
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
DATA_ROOT="${ADSB_DATA_ROOT:-数据集/ADS-B/Dataset}"
CHECKPOINT="${ADSB_GPCC_CHECKPOINT:-results/stage4/adsb_main_long_radcil_seed7_44470736/closedset_adsb_90known_strict.pth}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="results/stage6/adsb_gpcc_cil_seed7_${JOB_ID}"

mkdir -p results/stage6

"${PYTHON_BIN}" experiments/exp_adsb_mvacc_cil_strict.py \
    --data_root "${DATA_ROOT}" \
    --save_dir "${OUTPUT_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --initial_known_classes 90 \
    --round_size 10 \
    --num_rounds 3 \
    --seed 7 \
    --backbone adsb_long \
    --batch_size 128 \
    --test_batch_size 256 \
    --use_supcon \
    --supcon_weight 0.1 \
    --disable_cil_baselines \
    --disable_visualization \
    --discovery_backend gpcc \
    --cil_head_warmup_epochs 2 \
    --cil_joint_epochs 8 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0

"${PYTHON_BIN}" tools/stage6_adsb_gpcc_cil_seed7_report.py \
    --save-dir "${OUTPUT_DIR}" \
    --job-id "${JOB_ID}" \
    --output "results/stage6/STAGE6_ADSB_GPCC_CIL_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage6/stage6_adsb_gpcc_cil_seed7_summary_${JOB_ID}.json"
