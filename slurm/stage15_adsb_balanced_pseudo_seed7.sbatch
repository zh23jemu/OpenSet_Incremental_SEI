#!/bin/bash
# Stage 15：ADS-B 类均衡伪标签训练 seed7。
#
# 基于 cross_day + GPCC 路线，增加当前轮新伪类的反频率均衡采样和温和
# CE 均衡权重，用来验证新类被大簇/旧类回放压制是否是 R3 低分因素。

#SBATCH --job-name=osei-s15-adsb-balance
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage15/slurm_adsb_balanced_seed7_%j.out
#SBATCH --error=results/stage15/slurm_adsb_balanced_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
ADSB_ROOT="${ADSB_DATA_ROOT:-${DATA_ROOT}/数据集/ADS-B/Dataset}"
CHECKPOINT="${ADSB_CHECKPOINT:-${DATA_ROOT}/results/stage4/adsb_main_long_radcil_seed7_44470736/closedset_adsb_90known_strict.pth}"
SAVE_DIR="results/stage15/adsb_balanced_pseudo_seed7_${JOB_ID}"
mkdir -p results/stage15

"${PYTHON_BIN}" experiments/exp_adsb_mvacc_cil_strict.py \
    --data_root "${ADSB_ROOT}" \
    --save_dir "${SAVE_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --initial_known_classes 90 --round_size 10 --num_rounds 3 --seed 7 \
    --backbone adsb_long --batch_size 128 --test_batch_size 256 \
    --use_supcon --supcon_weight 0.1 \
    --disable_cil_baselines --disable_visualization \
    --discovery_backend gpcc \
    --cross_day_repr_adaptation \
    --cross_day_repr_epochs 2 \
    --cross_day_repr_batch_size 128 \
    --cross_day_repr_lr 1e-5 \
    --cross_day_repr_consistency_weight 0.5 \
    --cross_day_repr_coral_weight 0.05 \
    --radcil_balance_current_pseudo_classes \
    --radcil_current_class_balance_weight 0.5 \
    --radcil_current_class_balance_max_weight 2.0 \
    --cil_head_warmup_epochs 2 \
    --cil_joint_epochs 8 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0

"${PYTHON_BIN}" tools/stage15_adsb_balanced_pseudo_report.py \
    --save-dir "${SAVE_DIR}" \
    --output "results/stage15/STAGE15_ADSB_BALANCED_PSEUDO_SEED7_REPORT_${JOB_ID}.md"
