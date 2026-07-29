#!/bin/bash
# ADS-B target split 后端遗忘对照。
#
# 本作业固定已验证的 target split 发现前端，并打开内置 CIL baselines。
# 目标是判断 ADS-B 遗忘风险是否由 RADCIL 后端旧/新权衡主导，而不是继续
# 做 target split 小参数搜索。可在提交时用 sbatch --partition=目标分区
# 覆盖默认 gpu 分区。

#SBATCH --job-name=osei-stage4-adsb-ts-backend
#SBATCH --partition=gpu
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage4-adsb-ts-backend-%j.out
#SBATCH --error=results/slurm-osei-stage4-adsb-ts-backend-%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
mkdir -p results/stage4

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DATA_ROOT="${ADSB_DATA_ROOT:-数据集/ADS-B/Dataset}"
JOB_ID="${SLURM_JOB_ID:-manual}"
SEEDS="${ADSB_STAGE4_TARGET_BACKEND_SEEDS:-7,13,31}"
OUTPUT_PREFIX="${ADSB_STAGE4_TARGET_BACKEND_OUTPUT_PREFIX:-results/stage4/adsb_target_split_backend_baselines}"
PLAN_PATH="results/stage4/stage4_adsb_target_split_backend_plan_${JOB_ID}.json"
REPORT_PATH="results/stage4/STAGE4_ADSB_TARGET_SPLIT_BACKEND_REPORT_${JOB_ID}.md"
SUMMARY_PATH="results/stage4/stage4_adsb_target_split_backend_summary_${JOB_ID}.json"

"${PYTHON_BIN}" tools/stage4_adsb_target_split_backend_plan.py \
    --data-root "${DATA_ROOT}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --job-id "${JOB_ID}" \
    --output "${PLAN_PATH}"

checkpoint_for_seed() {
    case "$1" in
        7) echo "results/stage4/adsb_main_long_radcil_seed7_44470736/closedset_adsb_90known_strict.pth" ;;
        13) echo "results/stage4/adsb_main_long_radcil_seed13_44470736/closedset_adsb_90known_strict.pth" ;;
        31) echo "results/stage4/adsb_main_long_radcil_seed31_44467424/closedset_adsb_90known_strict.pth" ;;
        *) echo "不支持的 seed：$1" >&2; return 1 ;;
    esac
}

IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
for seed in "${SEED_ARRAY[@]}"; do
    seed="$(echo "${seed}" | tr -d '[:space:]')"
    [[ -z "${seed}" ]] && continue
    checkpoint="$(checkpoint_for_seed "${seed}")"
    save_dir="${OUTPUT_PREFIX}_seed${seed}_${JOB_ID}"
    echo "运行 ADS-B target split 后端对照：seed=${seed}, save_dir=${save_dir}"
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
        --enable_mvacc_target_split \
        --mvacc_target_split_clusters 10 \
        --mvacc_target_split_max_added 4 \
        --mvacc_target_split_silhouette 0.26 \
        --disable_visualization
done

"${PYTHON_BIN}" tools/stage4_adsb_target_split_backend_report.py \
    --job-id "${JOB_ID}" \
    --seeds "${SEEDS}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --baseline-summary-json "results/stage4/stage4_adsb_cil_baselines_summary_44517860.json" \
    --output "${REPORT_PATH}" \
    --summary-json "${SUMMARY_PATH}"

echo "ADS-B target split 后端遗忘对照完成：${REPORT_PATH}"
