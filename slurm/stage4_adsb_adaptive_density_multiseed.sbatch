#!/bin/bash
# ADS-B 无标签轮次自适应密度的三种子短验证。
#
# 每个 seed 复用已训练的长序列 checkpoint；后端固定为 old:new=2.0、
# replay weight=3.0。每轮只根据 discovery 侧无标签结构门控在 0.03/0.02
# 之间选择，并输出完整 CSV 审计。可在提交时用 --partition 覆盖默认分区。

#SBATCH --job-name=osei-stage4-adaptive-density
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage4-adaptive-density-%j.out
#SBATCH --error=results/slurm-osei-stage4-adaptive-density-%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
mkdir -p results/stage4

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DATA_ROOT="${ADSB_DATA_ROOT:-数据集/ADS-B/Dataset}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_PREFIX="results/stage4/adsb_adaptive_density"
PLAN_PATH="results/stage4/stage4_adsb_adaptive_density_plan_${JOB_ID}.json"
REPORT_PATH="results/stage4/STAGE4_ADSB_ADAPTIVE_DENSITY_REPORT_${JOB_ID}.md"
SUMMARY_PATH="results/stage4/stage4_adsb_adaptive_density_summary_${JOB_ID}.json"

"${PYTHON_BIN}" tools/stage4_adsb_adaptive_density_plan.py \
    --data-root "${DATA_ROOT}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --job-id "${JOB_ID}" \
    --output "${PLAN_PATH}"

run_seed() {
    local seed="$1"
    local checkpoint="$2"
    local save_dir="${OUTPUT_PREFIX}_seed${seed}_${JOB_ID}"

    echo "运行 ADS-B 自适应密度：seed=${seed}"
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
        --mvacc_min_cluster_ratio 0.03 \
        --mvacc_merge_threshold 0.74 \
        --mvacc_split_size_factor 1.75 \
        --mvacc_split_silhouette 0.30 \
        --enable_mvacc_adaptive_density \
        --mvacc_adaptive_candidate_ratio 0.02 \
        --mvacc_adaptive_max_added_clusters 2 \
        --mvacc_adaptive_min_silhouette_gain 0.04 \
        --mvacc_adaptive_max_confidence_drop 0.03 \
        --mvacc_adaptive_max_cv_increase 0.10 \
        --mvacc_adaptive_max_small_fraction_increase 0.10 \
        --mvacc_adaptive_min_partition_agreement 0.50 \
        --disable_visualization
}

run_seed 7 "results/stage4/adsb_main_long_radcil_seed7_44470736/closedset_adsb_90known_strict.pth"
run_seed 13 "results/stage4/adsb_main_long_radcil_seed13_44470736/closedset_adsb_90known_strict.pth"
run_seed 31 "results/stage4/adsb_main_long_radcil_seed31_44467424/closedset_adsb_90known_strict.pth"

"${PYTHON_BIN}" tools/stage4_adsb_adaptive_density_report.py \
    --plan "${PLAN_PATH}" \
    --job-id "${JOB_ID}" \
    --output "${REPORT_PATH}" \
    --summary-json "${SUMMARY_PATH}"

echo "ADS-B 自适应密度三种子完成：${REPORT_PATH}"
