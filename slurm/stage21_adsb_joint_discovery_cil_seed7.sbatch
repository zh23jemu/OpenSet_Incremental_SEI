#!/bin/bash
# Stage 21：ADS-B 联合 discovery-CIL seed7 验证。
#
# 每轮先用 GPCC 生成固定 K=10 的伪类并完成一次 CIL 更新，再用更新后的 Student
# 表征重新发现当前轮簇，并通过无标签原型 Hungarian 对齐回原伪类编号，最后再
# 训练一次。该闭环用于验证“聚类质量好，但伪类没有被增量后端吸收”的根因。

#SBATCH --job-name=osei-s21-adsb-joint
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage21/slurm_adsb_joint_discovery_cil_seed7_%j.out
#SBATCH --error=results/stage21/slurm_adsb_joint_discovery_cil_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
ADSB_ROOT="${ADSB_DATA_ROOT:-${DATA_ROOT}/数据集/ADS-B/Dataset}"
SAVE_DIR="results/stage21/adsb_joint_discovery_cil_seed7_${JOB_ID}"
CHECKPOINT="${SAVE_DIR}/closedset_adsb_90known_strict.pth"

mkdir -p results/stage21 "${SAVE_DIR}"

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
    --enable_joint_discovery_refinement \
    --disable_cil_baselines --disable_visualization \
    --cil_head_warmup_epochs 2 \
    --cil_joint_epochs 8 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0

"${PYTHON_BIN}" tools/stage18_adsb_backbone_cil_seed7_report.py \
    --save-dir "${SAVE_DIR}" \
    --output "results/stage21/STAGE21_ADSB_JOINT_DISCOVERY_CIL_SEED7_REPORT_${JOB_ID}.md"
