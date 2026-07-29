#!/bin/bash
# 阶段 5：LoRa seed7 BatchNorm 统计重校准诊断验证。
#
# 本作业用于验证 LoRa 低分是否主要来自跨天/跨采集日的表征统计漂移。
# BN 重校准只使用当前协议允许触达的数据：旧类 replay memory 与当前轮
# discovery/enrollment 样本；IQ_8-10 held-out evaluation 不参与 BN 统计估计、
# 参数选择或训练。可在提交时用 `sbatch --partition=目标分区 本脚本` 覆盖分区。

#SBATCH --job-name=osei-stage5-lora-bn-s7
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=results/slurm-osei-stage5-lora-bn-s7-%j.out
#SBATCH --error=results/slurm-osei-stage5-lora-bn-s7-%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="results/stage5/lora_bn_recalibration_seed7_${JOB_ID}"
CHECKPOINT="results/stage5/lora_representation_screen_seed7_44573249/supcon_rf_aug.pth"
LORA_NPZ="${LORA25_NPZ:-datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "缺少 LoRa IQ_7 预选 checkpoint：${CHECKPOINT}" >&2
    exit 2
fi

"${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
    --dataset_path "${LORA_NPZ}" \
    --save_dir "${OUTPUT_DIR}" \
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
    --radcil_bn_recalibration \
    --radcil_bn_recalibration_passes 1 \
    --mvacc_enforce_round_size \
    --mvacc_min_cluster_ratio 0.03 \
    --mvacc_min_samples 5

"${PYTHON_BIN}" tools/stage5_lora_bn_recalibration_report.py \
    --save-dir "${OUTPUT_DIR}" \
    --job-id "${JOB_ID}" \
    --output "results/stage5/STAGE5_LORA_BN_RECALIBRATION_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage5/stage5_lora_bn_recalibration_seed7_summary_${JOB_ID}.json"
