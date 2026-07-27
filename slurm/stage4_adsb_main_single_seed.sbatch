#!/bin/bash
# 阶段 4 ADS-B strict 单种子主流程 smoke test。
#
# 用途：
#   1. 复用阶段 0 已审计的 ADS-B strict loader；
#   2. 验证 ADS-B long-sequence backbone 能在 Slurm 上完整跑完 90+10x3；
#   3. 固定使用 WiSig 主后端的 old:new=2.0、replay weight=3.0 配置。
#
# 分区可在提交时覆盖，例如：
#   sbatch --partition=gpu --qos=normal --time=08:00:00 slurm/stage4_adsb_main_single_seed.sbatch

#SBATCH --job-name=osei-stage4-adsb
#SBATCH --partition=gpu
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage4-adsb-%j.out
#SBATCH --error=results/slurm-osei-stage4-adsb-%j.err

set -euo pipefail

module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

mkdir -p results/stage4

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DATA_ROOT="${ADSB_DATA_ROOT:-数据集/ADS-B/Dataset}"
JOB_ID="${SLURM_JOB_ID:-manual}"
SEED="${ADSB_STAGE4_SEED:-31}"
SAVE_DIR="results/stage4/adsb_main_long_radcil_seed${SEED}_${JOB_ID}"
PLAN_PATH="results/stage4/stage4_adsb_main_single_seed_plan_${JOB_ID}.json"

"${PYTHON_BIN}" tools/stage4_adsb_main_plan.py \
    --project-root "${PROJECT_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --save-dir "${SAVE_DIR}" \
    --seed "${SEED}" \
    --job-id "${JOB_ID}" \
    --output "${PLAN_PATH}"

echo "运行阶段 4 ADS-B strict 单种子流程：seed=${SEED}, save_dir=${SAVE_DIR}"
"${PYTHON_BIN}" experiments/exp_adsb_mvacc_cil_strict.py \
    --data_root "${DATA_ROOT}" \
    --save_dir "${SAVE_DIR}" \
    --initial_known_classes 90 \
    --round_size 10 \
    --num_rounds 3 \
    --seed "${SEED}" \
    --train_closedset \
    --backbone adsb_long \
    --epochs 30 \
    --batch_size 128 \
    --test_batch_size 256 \
    --disable_cil_baselines \
    --use_supcon \
    --supcon_weight 0.1 \
    --cil_head_warmup_epochs 2 \
    --cil_joint_epochs 8 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0 \
    --disable_visualization

echo "阶段 4 ADS-B strict 单种子流程完成：${PROJECT_ROOT}/${SAVE_DIR}"
