#!/bin/bash
# Stage 17：ADS-B 高置信伪标签注册 seed7 矩阵。
#
# 该任务验证低置信 discovery 样本是否污染新增分类头。每个新伪类内部按
# GPCC confidence 排序，只保留 top fraction 样本用于 imprint、当前轮训练和
# replay memory 更新；聚类指标仍基于全量 discovery 样本输出。

#SBATCH --job-name=osei-s17-adsb-reg
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage17/slurm_adsb_register_seed7_%j.out
#SBATCH --error=results/stage17/slurm_adsb_register_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
ADSB_ROOT="${ADSB_DATA_ROOT:-${DATA_ROOT}/数据集/ADS-B/Dataset}"
CHECKPOINT="${ADSB_CHECKPOINT:-${DATA_ROOT}/results/stage4/adsb_main_long_radcil_seed7_44470736/closedset_adsb_90known_strict.pth}"
ROOT_DIR="results/stage17/adsb_pseudo_register_seed7_${JOB_ID}"
mkdir -p results/stage17 "${ROOT_DIR}"

for FRACTION in 0.60 0.80; do
    SUFFIX="${FRACTION/./p}"
    SAVE_DIR="${ROOT_DIR}/top${SUFFIX}"
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
        --radcil_pseudo_register_top_fraction "${FRACTION}" \
        --radcil_pseudo_register_min_per_class 30 \
        --cil_head_warmup_epochs 2 \
        --cil_joint_epochs 8 \
        --cil_replay_weight 3.0 \
        --radcil_old_new_batch_ratio 2.0
done

"${PYTHON_BIN}" tools/stage17_adsb_pseudo_register_report.py \
    --root "${ROOT_DIR}" \
    --output "results/stage17/STAGE17_ADSB_PSEUDO_REGISTER_SEED7_REPORT_${JOB_ID}.md"
