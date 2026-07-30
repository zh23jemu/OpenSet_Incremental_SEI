#!/bin/bash
# Stage 18：ADS-B discovery backbone 重训后的 seed7 完整增量验证。
#
# 该任务只验证 discovery-only 已通过门槛的 supcon_w0p30_rfaug 候选。
# 流程是先用 Day1 已知类 train/validation 重训 closed-set backbone，再用 GPCC
# 固定每轮 10 簇，并接当前 ADS-B Long-RADCIL 后端。若 seed7 完整 CIL 不能超过
# 当前强对照，就不扩 seed13/31，避免继续消耗三种子训练资源。

#SBATCH --job-name=osei-s18-adsb-cil
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage18/slurm_adsb_backbone_cil_seed7_%j.out
#SBATCH --error=results/stage18/slurm_adsb_backbone_cil_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
ADSB_ROOT="${ADSB_DATA_ROOT:-${DATA_ROOT}/数据集/ADS-B/Dataset}"
SAVE_DIR="results/stage18/adsb_backbone_cil_seed7_${JOB_ID}"
CHECKPOINT="${SAVE_DIR}/closedset_adsb_90known_strict.pth"

mkdir -p results/stage18 "${SAVE_DIR}"

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
    --cil_head_warmup_epochs 2 \
    --cil_joint_epochs 8 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0

"${PYTHON_BIN}" tools/stage18_adsb_backbone_cil_seed7_report.py \
    --save-dir "${SAVE_DIR}" \
    --output "results/stage18/STAGE18_ADSB_BACKBONE_CIL_SEED7_REPORT_${JOB_ID}.md"
