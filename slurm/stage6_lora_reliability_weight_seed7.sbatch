#!/bin/bash
# 阶段 6：LoRa discovery 簇可靠性伪标签加权 seed7 二元验证。
#
# 只比较关闭/开启 discovery-side cluster reliability weighting，固定
# 表征、RADCIL 配比和协议目标簇数；不把 held-out eval 用于选择。

#SBATCH --job-name=osei-s6-lora-rel-s7
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage6-lora-rel-s7-%j.out
#SBATCH --error=results/slurm-osei-stage6-lora-rel-s7-%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
MATRIX_ROOT="results/stage6/lora_reliability_weight_seed7_${JOB_ID}"
CHECKPOINT="${LORA_RELIABILITY_CHECKPOINT:-results/stage5/lora_representation_screen_seed7_44573249/supcon_rf_aug.pth}"
LORA_NPZ="${LORA25_NPZ:-datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"

for VARIANT in reliability_off reliability_on; do
    EXTRA_ARGS=()
    if [[ "${VARIANT}" == "reliability_on" ]]; then
        EXTRA_ARGS+=(--pseudo_weight_use_cluster_reliability)
    fi
    "${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
        --dataset_path "${LORA_NPZ}" \
        --save_dir "${MATRIX_ROOT}/${VARIANT}" \
        --checkpoint "${CHECKPOINT}" \
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
        --pseudo_weight_cluster_reliability_floor 0.20 \
        --mvacc_enforce_round_size \
        --mvacc_min_cluster_ratio 0.03 \
        --mvacc_min_samples 5 \
        "${EXTRA_ARGS[@]}"
done

"${PYTHON_BIN}" tools/stage6_lora_reliability_weight_report.py \
    --matrix-root "${MATRIX_ROOT}" \
    --job-id "${JOB_ID}" \
    --output "results/stage6/STAGE6_LORA_RELIABILITY_WEIGHT_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage6/stage6_lora_reliability_weight_seed7_summary_${JOB_ID}.json"
