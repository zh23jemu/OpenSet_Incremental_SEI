#!/bin/bash
# 阶段 3 WiSig RADCIL + iCaRL 低置信回退三种子正式验证。
#
# seed7 已达到预注册扩展门槛后，本任务在 seed 7/13/31 上验证收益是否
# 稳定。除 confidence-gated exemplar fallback 外，其余前端、训练预算、
# replay 强度和 old:new batch ratio 均保持阶段 3 主方法配置。
#
# 可在提交时覆盖分区：
#   sbatch --partition=gpu --qos=normal --time=03:00:00 slurm/stage3_wisig_icarl_fallback_multiseed.sbatch

#SBATCH --job-name=osei-stage3-icarl-ms
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage3-icarl-ms-%j.out
#SBATCH --error=results/slurm-osei-stage3-icarl-ms-%j.err

set -euo pipefail

module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

mkdir -p results/stage3

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DATASET_PATH="${WISIG_DATASET_PATH:-数据集/WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl}"
JOB_ID="${SLURM_JOB_ID:-manual}"
SEEDS="${ICARL_FALLBACK_STAGE3_SEEDS:-7,13,31}"
OUTPUT_PREFIX="${ICARL_FALLBACK_STAGE3_OUTPUT_PREFIX:-results/stage3/wisig_icarl_fallback_multiseed}"
PLAN_PATH="results/stage3/stage3_wisig_icarl_fallback_multiseed_plan_${JOB_ID}.json"
REPORT_PATH="results/stage3/STAGE3_WISIG_ICARL_FALLBACK_MULTISEED_REPORT_${JOB_ID}.md"
SUMMARY_PATH="results/stage3/stage3_wisig_icarl_fallback_multiseed_summary_${JOB_ID}.json"

"${PYTHON_BIN}" tools/stage3_wisig_icarl_fallback_multiseed_plan.py \
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

    SAVE_DIR="${OUTPUT_PREFIX}_seed${seed}_${JOB_ID}"
    echo "运行阶段 3 WiSig iCaRL fallback：seed=${seed}, save_dir=${SAVE_DIR}"
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
        --radcil_old_new_batch_ratio 2.0 \
        --radcil_icarl_fallback \
        --radcil_icarl_fallback_candidates "0.45,0.55,0.65,0.75,0.85"
done

"${PYTHON_BIN}" tools/stage3_wisig_icarl_fallback_multiseed_report.py \
    --plan "${PLAN_PATH}" \
    --output "${REPORT_PATH}" \
    --summary-json "${SUMMARY_PATH}"

echo "阶段 3 WiSig iCaRL fallback 三种子验证完成：${PROJECT_ROOT}/results/stage3"
