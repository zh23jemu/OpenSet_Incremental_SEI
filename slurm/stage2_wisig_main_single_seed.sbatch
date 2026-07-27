#!/bin/bash
# 阶段 2 WiSig RADCIL 单种子主流程。
#
# 用途：
#   1. 固化阶段 1 多种子确认后的主后端 `ratio_2p0_replay_3p0`；
#   2. 使用稳定 MV-ACC 前端跑通 WiSig 10+10x3 完整三轮流程；
#   3. 将计划 JSON 与实验输出写入 results/stage2，便于后续客户汇报和三种子扩展。
#
# 分区可在提交时覆盖，例如：
#   sbatch --partition=gpuHz --qos=shortjobs --time=01:00:00 slurm/stage2_wisig_main_single_seed.sbatch

#SBATCH --job-name=osei-stage2-wisig
#SBATCH --partition=gpu
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage2-wisig-%j.out
#SBATCH --error=results/slurm-osei-stage2-wisig-%j.err

set -euo pipefail

module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

mkdir -p results/stage2

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DATASET_PATH="${WISIG_DATASET_PATH:-数据集/WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl}"
JOB_ID="${SLURM_JOB_ID:-manual}"
SEED="${RADCIL_STAGE2_SEED:-7}"
SAVE_DIR="results/stage2/wisig_main_ratio_2p0_replay_3p0_seed${SEED}_${JOB_ID}"
PLAN_PATH="results/stage2/stage2_wisig_main_single_seed_plan_${JOB_ID}.json"

"${PYTHON_BIN}" tools/stage2_wisig_main_plan.py \
    --project-root "${PROJECT_ROOT}" \
    --dataset-path "${DATASET_PATH}" \
    --save-dir "${SAVE_DIR}" \
    --seed "${SEED}" \
    --output "${PLAN_PATH}"

echo "运行阶段 2 WiSig 主流程：seed=${SEED}, save_dir=${SAVE_DIR}"
"${PYTHON_BIN}" experiments/exp_wisig_mvacc_cil_strict.py \
    --dataset_path "${DATASET_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --selected_rx_list 2 \
    --initial_known_classes 10 \
    --round_size 10 \
    --num_rounds 3 \
    --seed "${SEED}" \
    --train_closedset \
    --epochs 20 \
    --batch_size 128 \
    --test_batch_size 256 \
    --disable_cil_baselines \
    --use_supcon \
    --supcon_weight 0.1 \
    --cil_head_warmup_epochs 2 \
    --cil_joint_epochs 8 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0

"${PYTHON_BIN}" tools/stage2_wisig_main_report.py \
    --save-dir "${SAVE_DIR}" \
    --plan "${PLAN_PATH}" \
    --job-id "${JOB_ID}" \
    --output "results/stage2/STAGE2_WISIG_MAIN_SINGLE_SEED_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage2/stage2_wisig_main_single_seed_summary_${JOB_ID}.json"

echo "阶段 2 WiSig RADCIL 单种子主流程完成：${PROJECT_ROOT}/${SAVE_DIR}"
