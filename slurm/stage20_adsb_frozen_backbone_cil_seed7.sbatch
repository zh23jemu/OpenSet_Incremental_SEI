#!/bin/bash
# Stage 20：ADS-B 高质量 discovery backbone 冻结后的 seed7 增量验证。
#
# Stage 18/19 都证明 GPCC discovery 质量已经明显提升，但 R3 Overall 仍接近
# 0.498。这里直接关闭 CIL 阶段的 joint backbone 更新，只训练扩展分类头，
# 用来判断剩余损失是否来自增量阶段对 backbone 的二次破坏。该实验不新增
# 后验融合或阈值规则，结果可直接和 Stage 18/19 对照。

#SBATCH --job-name=osei-s20-adsb-frozen
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage20/slurm_adsb_frozen_backbone_seed7_%j.out
#SBATCH --error=results/stage20/slurm_adsb_frozen_backbone_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
ADSB_ROOT="${ADSB_DATA_ROOT:-${DATA_ROOT}/数据集/ADS-B/Dataset}"
SAVE_DIR="results/stage20/adsb_frozen_backbone_cil_seed7_${JOB_ID}"
CHECKPOINT="${SAVE_DIR}/closedset_adsb_90known_strict.pth"

mkdir -p results/stage20 "${SAVE_DIR}"

"${PYTHON_BIN}" experiments/exp_adsb_mvacc_cil_strict.py \
    --data_root "${ADSB_ROOT}" \
    --save_dir "${SAVE_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --train_closedset \
    --epochs 12 \
    --initial_known_classes 90 --round_size 10 --num_rounds 3 --seed 7 \
    --backbone adsb_long --batch_size 128 --test_batch_size 256 \
    --use_supcon --supcon_weight 0.30 \
    --discovery_backend gpcc \
    --disable_cil_baselines --disable_visualization \
    --cil_head_warmup_epochs 4 \
    --cil_joint_epochs 0 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0

"${PYTHON_BIN}" tools/stage18_adsb_backbone_cil_seed7_report.py \
    --save-dir "${SAVE_DIR}" \
    --output "results/stage20/STAGE20_ADSB_FROZEN_BACKBONE_CIL_SEED7_REPORT_${JOB_ID}.md"
