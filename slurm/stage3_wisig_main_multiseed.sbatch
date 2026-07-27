#!/bin/bash
# 阶段 3 WiSig RADCIL 正式三种子主实验。
#
# 用途：
#   1. 固定阶段 2 已跑通的 MV-ACC + ratio_2p0_replay_3p0 主组合；
#   2. 运行 seed 7/13/31，不根据中途结果继续调参；
#   3. 自动汇总 R1/R2/R3 均值和标准差，形成客户可汇报的正式 WiSig 主结果。
#
# 分区可在提交时覆盖，例如：
#   sbatch --partition=gpu --qos=normal --time=04:00:00 slurm/stage3_wisig_main_multiseed.sbatch

#SBATCH --job-name=osei-stage3-wisig
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage3-wisig-%j.out
#SBATCH --error=results/slurm-osei-stage3-wisig-%j.err

set -euo pipefail

module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

mkdir -p results/stage3

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DATASET_PATH="${WISIG_DATASET_PATH:-数据集/WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl}"
JOB_ID="${SLURM_JOB_ID:-manual}"
SEEDS="${RADCIL_STAGE3_SEEDS:-7,13,31}"
OUTPUT_PREFIX="results/stage3/wisig_main"
PLAN_PATH="results/stage3/stage3_wisig_main_multiseed_plan_${JOB_ID}.json"

"${PYTHON_BIN}" tools/stage3_wisig_main_multiseed_plan.py \
    --project-root "${PROJECT_ROOT}" \
    --dataset-path "${DATASET_PATH}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --job-id "${JOB_ID}" \
    --seeds "${SEEDS}" \
    --output "${PLAN_PATH}"

IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
for seed in "${SEED_ARRAY[@]}"; do
    seed="$(echo "${seed}" | tr -d '[:space:]')"
    if [[ -z "${seed}" ]]; then
        continue
    fi

    SAVE_DIR="${OUTPUT_PREFIX}_ratio_2p0_replay_3p0_seed${seed}_${JOB_ID}"
    echo "运行阶段 3 WiSig 正式主流程：seed=${seed}, save_dir=${SAVE_DIR}"
    "${PYTHON_BIN}" experiments/exp_wisig_mvacc_cil_strict.py \
        --dataset_path "${DATASET_PATH}" \
        --save_dir "${SAVE_DIR}" \
        --selected_rx_list 2 \
        --initial_known_classes 10 \
        --round_size 10 \
        --num_rounds 3 \
        --seed "${seed}" \
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
done

"${PYTHON_BIN}" tools/stage3_wisig_main_multiseed_report.py \
    --plan "${PLAN_PATH}" \
    --job-id "${JOB_ID}" \
    --output "results/stage3/STAGE3_WISIG_MAIN_MULTISEED_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage3/stage3_wisig_main_multiseed_summary_${JOB_ID}.json"

echo "阶段 3 WiSig RADCIL 正式三种子主实验完成：${PROJECT_ROOT}/results/stage3"
