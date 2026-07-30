#!/bin/bash
# Stage 14：ADS-B cross_day 表征 + target split 发现前端 seed7。
#
# 该实验测试表征适配与目标簇数补齐的结构组合；只跑一个 seed，
# 若 R3 Overall 超过 0.50 且 Old/New 不塌缩，再考虑三种子确认。

#SBATCH --job-name=osei-s14-adsb-cross-ts
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage14/slurm_adsb_cross_ts_seed7_%j.out
#SBATCH --error=results/stage14/slurm_adsb_cross_ts_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
ADSB_ROOT="${ADSB_DATA_ROOT:-${DATA_ROOT}/数据集/ADS-B/Dataset}"
CHECKPOINT="${ADSB_CHECKPOINT:-${DATA_ROOT}/results/stage4/adsb_main_long_radcil_seed7_44470736/closedset_adsb_90known_strict.pth}"
SAVE_DIR="results/stage14/adsb_cross_day_target_split_seed7_${JOB_ID}"
mkdir -p results/stage14

"${PYTHON_BIN}" experiments/exp_adsb_mvacc_cil_strict.py \
    --data_root "${ADSB_ROOT}" \
    --save_dir "${SAVE_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --initial_known_classes 90 --round_size 10 --num_rounds 3 --seed 7 \
    --backbone adsb_long --batch_size 128 --test_batch_size 256 \
    --use_supcon --supcon_weight 0.1 \
    --disable_cil_baselines --disable_visualization \
    --discovery_backend mvacc \
    --enable_mvacc_target_split \
    --mvacc_target_split_clusters 10 \
    --mvacc_target_split_max_added 4 \
    --mvacc_target_split_silhouette 0.26 \
    --cross_day_repr_adaptation \
    --cross_day_repr_epochs 2 \
    --cross_day_repr_batch_size 128 \
    --cross_day_repr_lr 1e-5 \
    --cross_day_repr_consistency_weight 0.5 \
    --cross_day_repr_coral_weight 0.05 \
    --cil_head_warmup_epochs 2 \
    --cil_joint_epochs 8 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0

"${PYTHON_BIN}" tools/stage14_adsb_cross_day_target_split_report.py \
    --save-dir "${SAVE_DIR}" \
    --output "results/stage14/STAGE14_ADSB_CROSS_DAY_TARGET_SPLIT_SEED7_REPORT_${JOB_ID}.md"
