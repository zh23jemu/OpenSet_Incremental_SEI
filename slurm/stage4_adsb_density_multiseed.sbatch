#!/bin/bash
# ADS-B min-cluster ratio 0.02 的配对三种子确认。
#
# seed7/13 各运行 baseline 与候选，seed31 复用 Job 44474297；所有运行复用
# 已训练长序列 checkpoint，只改变 MV-ACC min-cluster ratio。

#SBATCH --job-name=osei-stage4-density-ms
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage4-density-ms-%j.out
#SBATCH --error=results/slurm-osei-stage4-density-ms-%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
mkdir -p results/stage4

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DATA_ROOT="${ADSB_DATA_ROOT:-数据集/ADS-B/Dataset}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_PREFIX="results/stage4/adsb_density_multiseed"
PLAN_PATH="results/stage4/stage4_adsb_density_multiseed_plan_${JOB_ID}.json"
REPORT_PATH="results/stage4/STAGE4_ADSB_DENSITY_MULTISEED_REPORT_${JOB_ID}.md"
SUMMARY_PATH="results/stage4/stage4_adsb_density_multiseed_summary_${JOB_ID}.json"

"${PYTHON_BIN}" tools/stage4_adsb_density_multiseed_plan.py \
    --data-root "${DATA_ROOT}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --job-id "${JOB_ID}" \
    --output "${PLAN_PATH}"

run_pair() {
    local seed="$1"
    local checkpoint="$2"
    local variant="$3"
    local ratio="$4"
    local save_dir="${OUTPUT_PREFIX}_seed${seed}_${variant}_${JOB_ID}"

    echo "运行 ADS-B 密度配对：seed=${seed}, variant=${variant}"
    "${PYTHON_BIN}" experiments/exp_adsb_mvacc_cil_strict.py \
        --data_root "${DATA_ROOT}" \
        --save_dir "${save_dir}" \
        --checkpoint "${checkpoint}" \
        --initial_known_classes 90 \
        --round_size 10 \
        --num_rounds 3 \
        --seed "${seed}" \
        --backbone adsb_long \
        --batch_size 128 \
        --test_batch_size 256 \
        --disable_cil_baselines \
        --use_supcon \
        --supcon_weight 0.1 \
        --cil_head_warmup_epochs 2 \
        --cil_joint_epochs 8 \
        --cil_replay_weight 3.0 \
        --radcil_old_new_batch_ratio 2.0 \
        --disable_mvacc_calibration \
        --mvacc_min_cluster_ratio "${ratio}" \
        --mvacc_merge_threshold 0.74 \
        --mvacc_split_size_factor 1.75 \
        --mvacc_split_silhouette 0.30 \
        --disable_visualization
}

for seed in 7 13; do
    if [ "${seed}" = "7" ]; then
        checkpoint="results/stage4/adsb_main_long_radcil_seed7_44470736/closedset_adsb_90known_strict.pth"
    else
        checkpoint="results/stage4/adsb_main_long_radcil_seed13_44470736/closedset_adsb_90known_strict.pth"
    fi
    run_pair "${seed}" "${checkpoint}" baseline_locked 0.03
    run_pair "${seed}" "${checkpoint}" density_ratio_0p02 0.02
done

"${PYTHON_BIN}" tools/stage4_adsb_density_multiseed_report.py \
    --plan "${PLAN_PATH}" \
    --job-id "${JOB_ID}" \
    --output "${REPORT_PATH}" \
    --summary-json "${SUMMARY_PATH}"

echo "ADS-B 密度比例配对三种子完成：${REPORT_PATH}"
