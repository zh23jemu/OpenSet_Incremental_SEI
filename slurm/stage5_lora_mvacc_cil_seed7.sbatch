#!/bin/bash
# 阶段 5：LoRa25 strict MV-ACC-CIL seed7 正式单种子实验。
#
# Day1 IQ_1-6 训练、IQ_7 验证/校准、IQ_8-10 held-out 评估；
# Day2-4 IQ_1-7 discovery/enrollment、IQ_8-10 held-out 评估。
# 该作业只跑主方法，不运行额外 baseline，避免扩大单种子验证成本。
# 可在提交时覆盖分区：sbatch --partition=gpuHz slurm/stage5_lora_mvacc_cil_seed7.sbatch

#SBATCH --job-name=osei-stage5-lora-cil-s7
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage5-lora-cil-s7-%j.out
#SBATCH --error=results/slurm-osei-stage5-lora-cil-s7-%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
mkdir -p results/stage5

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="results/stage5/lora_mvacc_cil_seed7_${JOB_ID}"
LORA_NPZ="${LORA25_NPZ:-datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"

"${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
    --dataset_path "${LORA_NPZ}" \
    --save_dir "${OUTPUT_DIR}" \
    --checkpoint "${OUTPUT_DIR}/closedset_lora_day1_10known.pth" \
    --train_closedset \
    --seed 7 \
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
    --mvacc_min_cluster_ratio 0.03 \
    --mvacc_min_samples 5

"${PYTHON_BIN}" tools/stage5_lora_mvacc_cil_report.py \
    --save-dir "${OUTPUT_DIR}" \
    --job-id "${JOB_ID}" \
    --output "results/stage5/STAGE5_LORA_MVACC_CIL_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage5/stage5_lora_mvacc_cil_seed7_summary_${JOB_ID}.json"

echo "LoRa25 MV-ACC-CIL seed7 完成：${PROJECT_ROOT}/${OUTPUT_DIR}"
