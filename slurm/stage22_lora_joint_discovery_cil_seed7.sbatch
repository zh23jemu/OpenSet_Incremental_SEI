#!/bin/bash
# Stage 22：LoRa 联合 discovery-CIL seed7 验证。
#
# 每轮先用 GPCC/跨天适配完成发现和第一次 CIL，再用 Student 表征重新发现，
# 通过无标签类中心 Hungarian 对齐回原伪类编号，最后进行第二次 CIL。
# 该任务只验证结构性闭环是否能改善 LoRa 的旧新类失衡，不做相邻权重搜索。

#SBATCH --job-name=osei-s22-lora
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage22/slurm_lora_joint_discovery_cil_seed7_%j.out
#SBATCH --error=results/stage22/slurm_lora_joint_discovery_cil_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-${DATA_ROOT}/results/stage5/lora_representation_screen_seed7_44573249/supcon_rf_aug.pth}"
SAVE_DIR="results/stage22/lora_joint_discovery_cil_seed7_${JOB_ID}"

mkdir -p results/stage22 "${SAVE_DIR}"

"${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
    --dataset_profile lora25 \
    --dataset_path "${LORA_NPZ}" \
    --save_dir "${SAVE_DIR}" \
    --checkpoint "${LORA_CHECKPOINT}" \
    --seed 7 --initial_known_classes 10 --round_size 5 --num_rounds 3 \
    --use_supcon --supcon_weight 0.1 \
    --disable_cil_baselines --disable_visualization \
    --discovery_backend gpcc \
    --cross_day_repr_adaptation \
    --cross_day_repr_epochs 2 \
    --cross_day_repr_batch_size 128 \
    --cross_day_repr_lr 1e-5 \
    --cross_day_repr_consistency_weight 0.5 \
    --cross_day_repr_coral_weight 0.05 \
    --enable_joint_discovery_refinement \
    --cil_head_warmup_epochs 2 \
    --cil_joint_epochs 8 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0

"${PYTHON_BIN}" tools/stage5_lora_mvacc_cil_report.py \
    --save-dir "${SAVE_DIR}" \
    --job-id "${JOB_ID}" \
    --output "results/stage22/STAGE22_LORA_JOINT_DISCOVERY_CIL_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage22/stage22_lora_joint_discovery_cil_seed7_summary_${JOB_ID}.json"

echo "Stage 22 LoRa 联合 discovery-CIL seed7 完成。"
