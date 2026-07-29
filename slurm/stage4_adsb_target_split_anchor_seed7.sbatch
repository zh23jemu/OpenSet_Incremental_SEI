#!/bin/bash
# ADS-B target split + 训练期旧类原型锚定 seed7 消融。
#
# 本作业固定已验证的 target split 发现前端，只比较 RADCIL 训练期
# Teacher replay 类中心锚定权重。它不运行 CIL baselines，目标是在 1 小时
# 内判断该结构性遗忘控制候选是否值得扩展 seed13/31。可在提交时使用
# sbatch --partition=目标分区 覆盖默认 gpuHz 分区。

#SBATCH --job-name=osei-stage4-adsb-anchor
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage4-adsb-anchor-%j.out
#SBATCH --error=results/slurm-osei-stage4-adsb-anchor-%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
mkdir -p results/stage4

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DATA_ROOT="${ADSB_DATA_ROOT:-数据集/ADS-B/Dataset}"
JOB_ID="${SLURM_JOB_ID:-manual}"
WEIGHTS="${ADSB_STAGE4_ANCHOR_WEIGHTS:-0,0.10,0.25}"
OUTPUT_PREFIX="${ADSB_STAGE4_ANCHOR_OUTPUT_PREFIX:-results/stage4/adsb_target_split_anchor_seed7}"
PLAN_PATH="results/stage4/stage4_adsb_target_split_anchor_seed7_plan_${JOB_ID}.json"
REPORT_PATH="results/stage4/STAGE4_ADSB_TARGET_SPLIT_ANCHOR_SEED7_REPORT_${JOB_ID}.md"
SUMMARY_PATH="results/stage4/stage4_adsb_target_split_anchor_seed7_summary_${JOB_ID}.json"
CHECKPOINT="results/stage4/adsb_main_long_radcil_seed7_44470736/closedset_adsb_90known_strict.pth"

"${PYTHON_BIN}" tools/stage4_adsb_target_split_anchor_plan.py \
    --data-root "${DATA_ROOT}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --weights "${WEIGHTS}" \
    --job-id "${JOB_ID}" \
    --output "${PLAN_PATH}"

weight_tag() {
    echo "$1" | tr '.' 'p'
}

IFS=',' read -r -a WEIGHT_ARRAY <<< "${WEIGHTS}"
for weight in "${WEIGHT_ARRAY[@]}"; do
    weight="$(echo "${weight}" | tr -d '[:space:]')"
    [[ -z "${weight}" ]] && continue
    tag="$(weight_tag "${weight}")"
    save_dir="${OUTPUT_PREFIX}_w${tag}_${JOB_ID}"
    echo "运行 ADS-B target split 原型锚定：seed=7, weight=${weight}, save_dir=${save_dir}"
    "${PYTHON_BIN}" experiments/exp_adsb_mvacc_cil_strict.py \
        --data_root "${DATA_ROOT}" \
        --save_dir "${save_dir}" \
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
        --cil_head_warmup_epochs 2 \
        --cil_joint_epochs 8 \
        --cil_replay_weight 3.0 \
        --radcil_old_new_batch_ratio 2.0 \
        --radcil_prototype_anchor_weight "${weight}" \
        --disable_cil_baselines \
        --disable_mvacc_calibration \
        --mvacc_min_cluster_ratio 0.03 \
        --mvacc_merge_threshold 0.74 \
        --mvacc_split_size_factor 1.75 \
        --mvacc_split_silhouette 0.30 \
        --enable_mvacc_target_split \
        --mvacc_target_split_clusters 10 \
        --mvacc_target_split_max_added 4 \
        --mvacc_target_split_silhouette 0.26 \
        --disable_visualization
done

"${PYTHON_BIN}" tools/stage4_adsb_target_split_anchor_report.py \
    --job-id "${JOB_ID}" \
    --weights "${WEIGHTS}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --output "${REPORT_PATH}" \
    --summary-json "${SUMMARY_PATH}"

echo "ADS-B target split 原型锚定 seed7 消融完成：${REPORT_PATH}"
