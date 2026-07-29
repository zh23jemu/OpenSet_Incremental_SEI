#!/bin/bash
# 阶段 5：LoRa seed7 old-logit bias 诊断验证。
#
# 本作业用于回答“LoRa 低分是不是旧类 logits 被新增类头压低”。它复用
# IQ_7 预选的 supcon_rf_aug checkpoint，不重新训练初始模型；bias 候选
# 只通过 Day1 IQ_7 已知类验证集选择，IQ_8-10 held-out eval 只用于最终
# 一次性评估。可在提交时用 `sbatch --partition=目标分区 本脚本` 覆盖分区。

#SBATCH --job-name=osei-stage5-lora-bias-s7
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=results/slurm-osei-stage5-lora-bias-s7-%j.out
#SBATCH --error=results/slurm-osei-stage5-lora-bias-s7-%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="results/stage5/lora_old_logit_bias_seed7_${JOB_ID}"
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
    --radcil_old_logit_bias_candidates "0,0.5,1.0,1.5,2.0,3.0" \
    --mvacc_enforce_round_size \
    --mvacc_min_cluster_ratio 0.03 \
    --mvacc_min_samples 5

"${PYTHON_BIN}" tools/stage5_lora_old_logit_bias_report.py \
    --save-dir "${OUTPUT_DIR}" \
    --job-id "${JOB_ID}" \
    --output "results/stage5/STAGE5_LORA_OLD_LOGIT_BIAS_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage5/stage5_lora_old_logit_bias_seed7_summary_${JOB_ID}.json"
