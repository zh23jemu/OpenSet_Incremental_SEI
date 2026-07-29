#!/bin/bash
# 阶段 5：LoRa old-logit bias 三种子验证。
#
# 每个 seed 独立训练 Day1 closed-set checkpoint，并只用 Day1 IQ_7 选择旧类
# logit bias；IQ_8-10 held-out eval 只用于最终报告。该作业用于确认 seed7
# 的 old-logit bias 改善是否稳定，不继续原型锚定或 late-fusion 小参数搜索。

#SBATCH --job-name=osei-stage5-lora-bias-ms
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage5-lora-bias-ms-%j.out
#SBATCH --error=results/slurm-osei-stage5-lora-bias-ms-%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
JOB_ID="${SLURM_JOB_ID:-manual}"
MATRIX_ROOT="results/stage5/lora_old_logit_bias_multiseed_${JOB_ID}"
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
        --radcil_old_logit_bias_candidates "0,0.5,1.0,1.5,2.0,3.0" \
        --mvacc_enforce_round_size \
        --mvacc_min_cluster_ratio 0.03 \
        --mvacc_min_samples 5
done

"${PYTHON_BIN}" tools/stage5_lora_old_logit_bias_multiseed_report.py \
    --matrix-root "${MATRIX_ROOT}" \
    --seeds "7,13,31" \
    --job-id "${JOB_ID}" \
    --output "results/stage5/STAGE5_LORA_OLD_LOGIT_BIAS_MULTISEED_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage5/stage5_lora_old_logit_bias_multiseed_summary_${JOB_ID}.json"

echo "LoRa old-logit bias 三种子验证完成：${PROJECT_ROOT}/${MATRIX_ROOT}"
