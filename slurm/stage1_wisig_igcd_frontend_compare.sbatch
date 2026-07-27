#!/bin/bash
# 阶段 1：WiSig frozen embeddings 上的 IGCD strict 最小前端对比。
#
# 资源说明：
# - 默认使用 gpuHz + shortjobs，目标是在 1 小时内完成单种子短验证；
# - 输出写入 results/stage1/wisig_igcd_frontend_compare_${SLURM_JOB_ID}。

#SBATCH --job-name=osei-stage1-igcd
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage1-igcd-%j.out
#SBATCH --error=results/slurm-osei-stage1-igcd-%j.err

set -euo pipefail

module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

JOB_ID="${SLURM_JOB_ID:-manual}"
OUT_DIR="results/stage1/wisig_igcd_frontend_compare_${JOB_ID}"

.venv/bin/python tools/stage1_wisig_igcd_frontend_compare.py \
  --dataset_path "数据集/WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl" \
  --save_dir "${OUT_DIR}" \
  --selected_rx_list 2 \
  --initial_known_classes 10 \
  --round_size 10 \
  --num_rounds 3 \
  --seed 7 \
  --train_closedset \
  --epochs 8 \
  --batch_size 128 \
  --test_batch_size 256 \
  --use_supcon \
  --supcon_weight 0.1 \
  --disable_rf_augmentation
