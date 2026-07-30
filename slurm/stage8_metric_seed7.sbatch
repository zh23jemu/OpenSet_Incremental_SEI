#!/bin/bash
# Stage 8：ADS-B / LoRa 增量归一化代理度量 seed7 矩阵。
#
# 本作业只验证训练期特征几何重塑，不改 strict split，不读取 held-out
# 真值做选择。若 seed7 不过预注册门槛，直接归档为负消融，不扩三种子。

#SBATCH --job-name=osei-s8-metric-s7
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage8-metric-s7-%j.out
#SBATCH --error=results/slurm-osei-stage8-metric-s7-%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
WEIGHTS="${STAGE8_METRIC_WEIGHTS:-0,0.10,0.25}"
METRIC_SCALE="${STAGE8_METRIC_SCALE:-16.0}"
METRIC_MARGIN="${STAGE8_METRIC_MARGIN:-0.05}"
MATRIX_ROOT="results/stage8/incremental_metric_seed7_${JOB_ID}"
ADSB_DATA_ROOT="${ADSB_DATA_ROOT:-数据集/ADS-B/Dataset}"
ADSB_CHECKPOINT="${ADSB_CHECKPOINT:-results/stage4/adsb_main_long_radcil_seed7_44470736/closedset_adsb_90known_strict.pth}"
LORA_NPZ="${LORA25_NPZ:-datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-results/stage5/lora_representation_screen_seed7_44573249/supcon_rf_aug.pth}"

"${PYTHON_BIN}" tools/stage8_metric_plan.py \
    --weights "${WEIGHTS}" \
    --metric-scale "${METRIC_SCALE}" \
    --metric-margin "${METRIC_MARGIN}" \
    --output "results/stage8/stage8_incremental_metric_seed7_plan_${JOB_ID}.json"

weight_tag() {
    echo "$1" | tr '.' 'p'
}

IFS=',' read -r -a WEIGHT_ARRAY <<< "${WEIGHTS}"
for weight in "${WEIGHT_ARRAY[@]}"; do
    weight="$(echo "${weight}" | tr -d '[:space:]')"
    [[ -z "${weight}" ]] && continue
    tag="$(weight_tag "${weight}")"

    echo "运行 ADS-B metric weight=${weight}"
    "${PYTHON_BIN}" experiments/exp_adsb_mvacc_cil_strict.py \
        --data_root "${ADSB_DATA_ROOT}" \
        --save_dir "${MATRIX_ROOT}/adsb/metric_w${tag}" \
        --checkpoint "${ADSB_CHECKPOINT}" \
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
        --radcil_metric_weight "${weight}" \
        --radcil_metric_scale "${METRIC_SCALE}" \
        --radcil_metric_margin "${METRIC_MARGIN}" \
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

    echo "运行 LoRa metric weight=${weight}"
    "${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
        --dataset_path "${LORA_NPZ}" \
        --save_dir "${MATRIX_ROOT}/lora/metric_w${tag}" \
        --checkpoint "${LORA_CHECKPOINT}" \
        --seed 7 \
        --initial_known_classes 10 \
        --round_size 5 \
        --num_rounds 3 \
        --use_supcon \
        --supcon_weight 0.1 \
        --disable_cil_baselines \
        --disable_visualization \
        --cil_head_warmup_epochs 2 \
        --cil_joint_epochs 8 \
        --cil_replay_weight 3.0 \
        --radcil_old_new_batch_ratio 2.0 \
        --radcil_metric_weight "${weight}" \
        --radcil_metric_scale "${METRIC_SCALE}" \
        --radcil_metric_margin "${METRIC_MARGIN}" \
        --mvacc_enforce_round_size \
        --mvacc_min_cluster_ratio 0.03 \
        --mvacc_min_samples 5
done

"${PYTHON_BIN}" tools/stage8_metric_report.py \
    --matrix-root "${MATRIX_ROOT}" \
    --weights "${WEIGHTS}" \
    --job-id "${JOB_ID}" \
    --output "results/stage8/STAGE8_INCREMENTAL_METRIC_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage8/stage8_incremental_metric_seed7_summary_${JOB_ID}.json"

echo "Stage 8 metric seed7 完成。"
