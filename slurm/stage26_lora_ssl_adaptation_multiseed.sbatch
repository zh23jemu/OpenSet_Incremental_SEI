#!/bin/bash
# Stage 26：LoRa SSL + cross-day 三种子确认。
#
# 只扩展 seed7 已通过门槛的 ssl_plus_cross_day 结构候选。每个 seed 都重新
# 训练 Day1 closed-set backbone，再做每轮 discovery 前 SSL 适配、cross-day
# 适配、GPCC 和联合 discovery-CIL，避免复用 seed7 checkpoint。

#SBATCH --job-name=osei-s26-lora-ms
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage26/slurm_lora_ssl_adaptation_multiseed_%j.out
#SBATCH --error=results/stage26/slurm_lora_ssl_adaptation_multiseed_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
SEEDS="${STAGE26_LORA_SEEDS:-7,13,31}"

mkdir -p results/stage26
IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"

for seed in "${SEED_ARRAY[@]}"; do
    seed="$(echo "${seed}" | tr -d '[:space:]')"
    SAVE_DIR="results/stage26/lora_ssl_plus_cross_day_seed${seed}_${JOB_ID}"
    CHECKPOINT="${SAVE_DIR}/closedset_lora25_seed${seed}.pth"
    mkdir -p "${SAVE_DIR}"
    echo "运行 Stage 26 LoRa SSL + cross-day：seed=${seed}"
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
        --discovery_backend gpcc \
        --lora_ssl_adaptation \
        --lora_ssl_epochs 3 \
        --lora_ssl_batch_size 128 \
        --lora_ssl_lr 1e-5 \
        --lora_ssl_ce_weight 1.0 \
        --lora_ssl_instance_weight 0.5 \
        --lora_ssl_teacher_weight 0.2 \
        --lora_ssl_temperature 0.2 \
        --cross_day_repr_adaptation \
        --cross_day_repr_epochs 2 \
        --cross_day_repr_batch_size 128 \
        --cross_day_repr_lr 1e-5 \
        --cross_day_repr_consistency_weight 0.5 \
        --cross_day_repr_coral_weight 0.05 \
        --enable_joint_discovery_refinement \
        --enable_lora_recording_eval \
        --cil_head_warmup_epochs 2 \
        --cil_joint_epochs 8 \
        --cil_replay_weight 3.0 \
        --radcil_old_new_batch_ratio 2.0
done

"${PYTHON_BIN}" tools/stage26_lora_ssl_adaptation_multiseed_report.py \
    --root results/stage26 \
    --job-id "${JOB_ID}" \
    --seeds "${SEEDS}" \
    --output "results/stage26/STAGE26_LORA_SSL_ADAPTATION_MULTISEED_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage26/stage26_lora_ssl_adaptation_multiseed_summary_${JOB_ID}.json"

echo "Stage 26 LoRa SSL + cross-day 三种子确认完成。"
