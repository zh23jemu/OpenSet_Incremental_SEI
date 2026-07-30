#!/bin/bash
# Stage 16：ADS-B 旧/新双分支训练 seed7 矩阵。
#
# 这个短任务继续沿用 cross_day + GPCC 的发现前端，只在 RADCIL 增量训练阶段
# 增加旧/新分支判别损失。旧分支表示当前轮之前所有已注册类，新分支表示当前轮
# 刚发现的新伪类。该损失只使用训练侧伪标签和 replay 标签，不读取未知真值调参。

#SBATCH --job-name=osei-s16-adsb-branch
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage16/slurm_adsb_branch_seed7_%j.out
#SBATCH --error=results/stage16/slurm_adsb_branch_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
ADSB_ROOT="${ADSB_DATA_ROOT:-${DATA_ROOT}/数据集/ADS-B/Dataset}"
CHECKPOINT="${ADSB_CHECKPOINT:-${DATA_ROOT}/results/stage4/adsb_main_long_radcil_seed7_44470736/closedset_adsb_90known_strict.pth}"
ROOT_DIR="results/stage16/adsb_old_new_branch_seed7_${JOB_ID}"
mkdir -p results/stage16 "${ROOT_DIR}"

for WEIGHT in 0.30 0.70; do
    SUFFIX="${WEIGHT/./p}"
    SAVE_DIR="${ROOT_DIR}/branch_w${SUFFIX}"
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
        --radcil_old_new_dual_branch \
        --radcil_old_new_branch_weight "${WEIGHT}" \
        --cil_head_warmup_epochs 2 \
        --cil_joint_epochs 8 \
        --cil_replay_weight 3.0 \
        --radcil_old_new_batch_ratio 2.0
done

"${PYTHON_BIN}" tools/stage16_adsb_old_new_branch_report.py \
    --root "${ROOT_DIR}" \
    --output "results/stage16/STAGE16_ADSB_OLD_NEW_BRANCH_SEED7_REPORT_${JOB_ID}.md"
