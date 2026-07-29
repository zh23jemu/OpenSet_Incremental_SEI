#!/bin/bash
# 阶段 5：LoRa BatchNorm 统计重校准三种子验证。
#
# seed7 已显示 BN 重校准能在不牺牲新类的前提下略微改善 Overall/Old。
# 本作业重新训练 seed 7/13/31 的 Day1 closed-set checkpoint，并在每轮
# discovery 前、student 训练后用协议允许的数据刷新 BN running mean/var。
# IQ_8-10 held-out evaluation 只用于最终报告，不参与 BN 统计估计或选择。

#SBATCH --job-name=osei-stage5-lora-bn-ms
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage5-lora-bn-ms-%j.out
#SBATCH --error=results/slurm-osei-stage5-lora-bn-ms-%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
JOB_ID="${SLURM_JOB_ID:-manual}"
MATRIX_ROOT="results/stage5/lora_bn_recalibration_multiseed_${JOB_ID}"
LORA_NPZ="${LORA25_NPZ:-datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
SEEDS=(7 13 31)

for SEED in "${SEEDS[@]}"; do
    OUT_DIR="${MATRIX_ROOT}/seed${SEED}"
    "${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
        --dataset_path "${LORA_NPZ}" \
        --save_dir "${OUT_DIR}" \
        --checkpoint "${OUT_DIR}/closedset_lora_day1_10known.pth" \
        --train_closedset \
        --seed "${SEED}" \
        --initial_known_classes 10 \
        --round_size 5 \
        --num_rounds 3 \
        --epochs 20 \
        --use_supcon \
        --supcon_weight 0.1 \
        --disable_cil_baselines \
        --disable_visualization \
        --cil_head_warmup_epochs 2 \
        --cil_joint_epochs 8 \
        --cil_replay_weight 3.0 \
        --radcil_old_new_batch_ratio 2.0 \
        --radcil_bn_recalibration \
        --radcil_bn_recalibration_passes 1 \
        --mvacc_enforce_round_size \
        --mvacc_min_cluster_ratio 0.03 \
        --mvacc_min_samples 5
done

"${PYTHON_BIN}" tools/stage5_lora_bn_recalibration_multiseed_report.py \
    --matrix-root "${MATRIX_ROOT}" \
    --seeds "7,13,31" \
    --job-id "${JOB_ID}" \
    --output "results/stage5/STAGE5_LORA_BN_RECALIBRATION_MULTISEED_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage5/stage5_lora_bn_recalibration_multiseed_summary_${JOB_ID}.json"

echo "LoRa BN 重校准三种子验证完成：${PROJECT_ROOT}/${MATRIX_ROOT}"
