#!/bin/bash
# Stage 19：ADS-B 初始表征教师蒸馏 seed7 结构验证。
#
# Stage 18 证明重训 backbone 能提升 GPCC 聚类，但完整 CIL 在 R3 又掉回约 0.498。
# 本任务保留 Stage 18 的 supcon_w0p30_rfaug backbone 配方，只新增一个长期冻结的
# 初始表征教师，约束高置信新样本和 replay 的特征方向，验证增量训练是否重新破坏
# discovery 阶段形成的表征结构。

#SBATCH --job-name=osei-s19-adsb-ft
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage19/slurm_adsb_initial_feature_teacher_seed7_%j.out
#SBATCH --error=results/stage19/slurm_adsb_initial_feature_teacher_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
ADSB_ROOT="${ADSB_DATA_ROOT:-${DATA_ROOT}/数据集/ADS-B/Dataset}"
SAVE_DIR="results/stage19/adsb_initial_feature_teacher_seed7_${JOB_ID}"
CHECKPOINT="${SAVE_DIR}/closedset_adsb_90known_strict.pth"

mkdir -p results/stage19 "${SAVE_DIR}"

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
    --radcil_old_new_batch_ratio 2.0 \
    --radcil_initial_feature_distill_weight 0.50

"${PYTHON_BIN}" tools/stage18_adsb_backbone_cil_seed7_report.py \
    --save-dir "${SAVE_DIR}" \
    --output "results/stage19/STAGE19_ADSB_INITIAL_FEATURE_TEACHER_SEED7_REPORT_${JOB_ID}.md"
