#!/bin/bash
# 阶段 3 WiSig hybrid RADCIL + DOI-memory 正式三种子验证。
# Job 44465809 已通过 seed7 门槛，本脚本锁定相同参数运行 seed 7/13/31。

#SBATCH --job-name=osei-stage3-hybrid-doi-ms
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage3-hybrid-doi-ms-%j.out
#SBATCH --error=results/slurm-osei-stage3-hybrid-doi-ms-%j.err

set -euo pipefail

module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

mkdir -p results/stage3

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DATASET_PATH="${WISIG_DATASET_PATH:-数据集/WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl}"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_PREFIX="results/stage3/wisig_hybrid_doi_multiseed"
PLAN_PATH="results/stage3/stage3_wisig_hybrid_doi_multiseed_plan_${JOB_ID}.json"
REPORT_PATH="results/stage3/STAGE3_WISIG_HYBRID_DOI_MULTISEED_REPORT_${JOB_ID}.md"
SUMMARY_PATH="results/stage3/stage3_wisig_hybrid_doi_multiseed_summary_${JOB_ID}.json"

"${PYTHON_BIN}" tools/stage3_wisig_hybrid_doi_multiseed_plan.py \
    --project-root "${PROJECT_ROOT}" \
    --dataset-path "${DATASET_PATH}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --job-id "${JOB_ID}" \
    --output "${PLAN_PATH}"

for seed in 7 13 31; do
    SAVE_DIR="${OUTPUT_PREFIX}_seed${seed}_${JOB_ID}"
    echo "运行 hybrid DOI 正式验证：seed=${seed}, save_dir=${SAVE_DIR}"
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
        --disable_visualization \
        --disable_cil_baselines \
        --use_supcon \
        --supcon_weight 0.1 \
        --cil_head_warmup_epochs 2 \
        --cil_joint_epochs 8 \
        --cil_replay_weight 3.0 \
        --radcil_old_new_batch_ratio 2.0 \
        --radcil_doi_fusion_weight 0.30 \
        --radcil_doi_align_lambda 0.30 \
        --radcil_doi_prototype_temperature 0.10
done

"${PYTHON_BIN}" tools/stage3_wisig_main_multiseed_report.py \
    --plan "${PLAN_PATH}" \
    --job-id "${JOB_ID}" \
    --output "${REPORT_PATH}" \
    --summary-json "${SUMMARY_PATH}"

echo "阶段 3 WiSig hybrid DOI 正式三种子验证完成。"
