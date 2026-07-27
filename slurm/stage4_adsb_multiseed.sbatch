#!/bin/bash
# 阶段 4 ADS-B Long-RADCIL 正式三种子扩展。
#
# seed31 已由 Job 44467424 完成，本脚本只运行 seed7/13，随后复用 seed31
# 生成三种子均值与标准差。提交时可用 --partition 覆盖默认 gpuHz。

#SBATCH --job-name=osei-stage4-adsb-ms
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage4-adsb-ms-%j.out
#SBATCH --error=results/slurm-osei-stage4-adsb-ms-%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
mkdir -p results/stage4

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DATA_ROOT="${ADSB_DATA_ROOT:-数据集/ADS-B/Dataset}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_PREFIX="results/stage4/adsb_main_long_radcil"
REUSED_SEED31_DIR="results/stage4/adsb_main_long_radcil_seed31_44467424"
PLAN_PATH="results/stage4/stage4_adsb_multiseed_plan_${JOB_ID}.json"
REPORT_PATH="results/stage4/STAGE4_ADSB_MULTISEED_REPORT_${JOB_ID}.md"
SUMMARY_PATH="results/stage4/stage4_adsb_multiseed_summary_${JOB_ID}.json"

"${PYTHON_BIN}" tools/stage4_adsb_multiseed_plan.py \
    --project-root "${PROJECT_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --job-id "${JOB_ID}" \
    --reused-seed31-dir "${REUSED_SEED31_DIR}" \
    --output "${PLAN_PATH}"

for seed in 7 13; do
    SAVE_DIR="${OUTPUT_PREFIX}_seed${seed}_${JOB_ID}"
    echo "运行 ADS-B 正式扩展：seed=${seed}, save_dir=${SAVE_DIR}"
    "${PYTHON_BIN}" experiments/exp_adsb_mvacc_cil_strict.py \
        --data_root "${DATA_ROOT}" \
        --save_dir "${SAVE_DIR}" \
        --initial_known_classes 90 \
        --round_size 10 \
        --num_rounds 3 \
        --seed "${seed}" \
        --train_closedset \
        --backbone adsb_long \
        --epochs 30 \
        --batch_size 128 \
        --test_batch_size 256 \
        --disable_cil_baselines \
        --use_supcon \
        --supcon_weight 0.1 \
        --cil_head_warmup_epochs 2 \
        --cil_joint_epochs 8 \
        --cil_replay_weight 3.0 \
        --radcil_old_new_batch_ratio 2.0 \
        --disable_visualization
done

"${PYTHON_BIN}" tools/stage4_adsb_multiseed_report.py \
    --plan "${PLAN_PATH}" \
    --job-id "${JOB_ID}" \
    --output "${REPORT_PATH}" \
    --summary-json "${SUMMARY_PATH}"

echo "ADS-B 阶段 4 正式三种子扩展完成：${REPORT_PATH}"
