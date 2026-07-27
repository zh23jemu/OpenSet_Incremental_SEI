#!/bin/bash
# ADS-B seed31 发现前端单因素短消融。
#
# 四个变体复用 Job 44467424 的长序列闭集 checkpoint，不重复训练 backbone；
# Long-RADCIL 后端完全固定。提交时可用 --partition 覆盖默认 gpuHz。

#SBATCH --job-name=osei-stage4-disc-ab
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage4-disc-ab-%j.out
#SBATCH --error=results/slurm-osei-stage4-disc-ab-%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
mkdir -p results/stage4

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DATA_ROOT="${ADSB_DATA_ROOT:-数据集/ADS-B/Dataset}"
JOB_ID="${SLURM_JOB_ID:-manual}"
CHECKPOINT="results/stage4/adsb_main_long_radcil_seed31_44467424/closedset_adsb_90known_strict.pth"
OUTPUT_PREFIX="results/stage4/adsb_discovery_ablation_seed31"
PLAN_PATH="results/stage4/stage4_adsb_discovery_ablation_plan_${JOB_ID}.json"
REPORT_PATH="results/stage4/STAGE4_ADSB_DISCOVERY_ABLATION_REPORT_${JOB_ID}.md"
SUMMARY_PATH="results/stage4/stage4_adsb_discovery_ablation_summary_${JOB_ID}.json"

"${PYTHON_BIN}" tools/stage4_adsb_discovery_ablation_plan.py \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --job-id "${JOB_ID}" \
    --output "${PLAN_PATH}"

run_variant() {
    local name="$1"
    local ratio="$2"
    local merge_threshold="$3"
    local split_factor="$4"
    local split_silhouette="$5"
    local save_dir="${OUTPUT_PREFIX}_${name}_${JOB_ID}"

    echo "运行 ADS-B 发现消融：${name}"
    "${PYTHON_BIN}" experiments/exp_adsb_mvacc_cil_strict.py \
        --data_root "${DATA_ROOT}" \
        --save_dir "${save_dir}" \
        --checkpoint "${CHECKPOINT}" \
        --initial_known_classes 90 \
        --round_size 10 \
        --num_rounds 3 \
        --seed 31 \
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
        --mvacc_merge_threshold "${merge_threshold}" \
        --mvacc_split_size_factor "${split_factor}" \
        --mvacc_split_silhouette "${split_silhouette}" \
        --disable_visualization
}

run_variant baseline_locked 0.03 0.74 1.75 0.30
run_variant density_ratio_0p02 0.02 0.74 1.75 0.30
run_variant merge_threshold_0p86 0.03 0.86 1.75 0.30
run_variant split_relaxed 0.03 0.74 1.50 0.20

"${PYTHON_BIN}" tools/stage4_adsb_discovery_ablation_report.py \
    --plan "${PLAN_PATH}" \
    --job-id "${JOB_ID}" \
    --output "${REPORT_PATH}" \
    --summary-json "${SUMMARY_PATH}"

echo "ADS-B 发现前端单因素消融完成：${REPORT_PATH}"
