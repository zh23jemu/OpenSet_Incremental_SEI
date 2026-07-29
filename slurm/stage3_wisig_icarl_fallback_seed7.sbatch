#!/bin/bash
# 阶段 3 WiSig RADCIL + iCaRL 低置信回退 seed7 短验证。
#
# 该任务保持 MV-ACC 前端和 ratio_2p0_replay_3p0 网络训练配置不变，
# 仅新增 confidence-gated exemplar fallback。阈值只用 Day1 validation
# 选择，held-out evaluation 真值只用于最终事后指标。
#
# 可在提交时覆盖分区：
#   sbatch --partition=gpu --qos=normal --time=02:00:00 slurm/stage3_wisig_icarl_fallback_seed7.sbatch

#SBATCH --job-name=osei-stage3-icarl-fb
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage3-icarl-fb-%j.out
#SBATCH --error=results/slurm-osei-stage3-icarl-fb-%j.err

set -euo pipefail

module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

mkdir -p results/stage3

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DATASET_PATH="${WISIG_DATASET_PATH:-数据集/WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl}"
JOB_ID="${SLURM_JOB_ID:-manual}"
SAVE_DIR="results/stage3/wisig_icarl_fallback_seed7_${JOB_ID}"
PLAN_PATH="results/stage3/stage3_wisig_icarl_fallback_seed7_plan_${JOB_ID}.json"
REPORT_PATH="results/stage3/STAGE3_WISIG_ICARL_FALLBACK_SEED7_REPORT_${JOB_ID}.md"
SUMMARY_PATH="results/stage3/stage3_wisig_icarl_fallback_seed7_summary_${JOB_ID}.json"

"${PYTHON_BIN}" tools/stage3_wisig_icarl_fallback_plan.py \
    --project-root "${PROJECT_ROOT}" \
    --dataset-path "${DATASET_PATH}" \
    --save-dir "${SAVE_DIR}" \
    --job-id "${JOB_ID}" \
    --output "${PLAN_PATH}"

"${PYTHON_BIN}" experiments/exp_wisig_mvacc_cil_strict.py \
    --dataset_path "${DATASET_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --selected_rx_list 2 \
    --initial_known_classes 10 \
    --round_size 10 \
    --num_rounds 3 \
    --seed 7 \
    --train_closedset \
    --epochs 20 \
    --batch_size 128 \
    --test_batch_size 256 \
    --disable_visualization \
    --disable_cil_baselines \
    --use_supcon \
    --supcon_weight 0.1 \
    --cil_head_warmup_epochs 2 \
    --cil_joint_epochs 8 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0 \
    --radcil_icarl_fallback \
    --radcil_icarl_fallback_candidates "0.45,0.55,0.65,0.75,0.85"

"${PYTHON_BIN}" tools/stage3_wisig_icarl_fallback_report.py \
    --plan "${PLAN_PATH}" \
    --output "${REPORT_PATH}" \
    --summary-json "${SUMMARY_PATH}"

echo "阶段 3 WiSig iCaRL fallback seed7 短验证完成：${SAVE_DIR}"
