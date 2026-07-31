#!/bin/bash
# Stage 24：LoRa recording-level GPCC 三种子确认。
#
# seed7 中 recording-level GPCC 提升了 Overall/Old 并显著降低遗忘，
# 但 New 有下降。本任务不继续调参数，只按公平三种子口径验证该结构
# 是否稳定优于 256 基线。每个 seed 单独重训 20 epoch closed-set backbone。

#SBATCH --job-name=osei-s24-lora-rec-ms
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage24/slurm_lora_recording_gpcc_multiseed_%j.out
#SBATCH --error=results/stage24/slurm_lora_recording_gpcc_multiseed_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
SEEDS="${STAGE24_LORA_SEEDS:-7,13,31}"

cd "${PROJECT_ROOT}"
mkdir -p results/stage24
IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"

for seed in "${SEED_ARRAY[@]}"; do
    seed="$(echo "${seed}" | tr -d '[:space:]')"
    SAVE_DIR="results/stage24/lora_recording_gpcc_seed${seed}_${JOB_ID}"
    CHECKPOINT="${SAVE_DIR}/closedset_lora25_seed${seed}.pth"
    mkdir -p "${SAVE_DIR}"
    echo "运行 Stage 24 LoRa Recording-GPCC：seed=${seed}"
    "${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
        --dataset_profile lora25 \
        --dataset_path "${LORA_NPZ}" \
        --save_dir "${SAVE_DIR}" \
        --checkpoint "${CHECKPOINT}" \
        --train_closedset \
        --epochs 20 \
        --seed "${seed}" --initial_known_classes 10 --round_size 5 --num_rounds 3 \
        --use_supcon --supcon_weight 0.1 \
        --disable_cil_baselines --disable_visualization \
        --discovery_backend gpcc_recording \
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
done

"${PYTHON_BIN}" tools/stage24_lora_recording_gpcc_multiseed_report.py \
    --root results/stage24 \
    --job-id "${JOB_ID}" \
    --seeds "${SEEDS}" \
    --output "results/stage24/STAGE24_LORA_RECORDING_GPCC_MULTISEED_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage24/stage24_lora_recording_gpcc_multiseed_summary_${JOB_ID}.json"

echo "Stage 24 LoRa Recording-GPCC 三种子实验完成。"
