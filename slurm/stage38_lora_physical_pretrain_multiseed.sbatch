#!/bin/bash
# Stage 38：LoRa 物理域自监督预训练三种子确认。
#
# 每个 seed 都重新训练一个 LoRa Chirp 初始 checkpoint；checkpoint 训练只使用
# Day1 IQ_1-6 训练样本和 IQ_7 验证样本，不读取 IQ_8-10 held-out eval。
# 后续 discovery/CIL 配置固定为 Stage 37/38 seed7 口径，用于验证 seed7
# 正信号是否能稳定扩展到 seed7/13/31。

#SBATCH --job-name=osei-s38-lora-ms
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage38/slurm_lora_physical_pretrain_multiseed_%j.out
#SBATCH --error=results/stage38/slurm_lora_physical_pretrain_multiseed_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
SEEDS="${STAGE38_LORA_SEEDS:-7,13,31}"
ROOT="results/stage38"

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}"
IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"

"${PYTHON_BIN}" tools/stage38_lora_physical_pretrain_plan.py \
    --output "${ROOT}/stage38_lora_physical_pretrain_multiseed_plan_${JOB_ID}.json"

for seed in "${SEED_ARRAY[@]}"; do
    seed="$(echo "${seed}" | tr -d '[:space:]')"
    CHECKPOINT="${ROOT}/lora_physical_pretrain_checkpoint_seed${seed}_${JOB_ID}.pth"
    CKPT_SUMMARY="${ROOT}/stage38_lora_physical_pretrain_checkpoint_summary_seed${seed}_${JOB_ID}.json"
    SAVE_DIR="${ROOT}/lora_physical_pretrain_seed${seed}_${JOB_ID}"

    echo "运行 Stage 38 LoRa physical pretrain：seed=${seed}"
    "${PYTHON_BIN}" tools/stage38_lora_physical_pretrain.py \
        --dataset-path "${LORA_NPZ}" \
        --output "${CHECKPOINT}" \
        --summary-json "${CKPT_SUMMARY}" \
        --seed "${seed}" \
        --epochs 20 \
        --batch-size 128 \
        --lr 1e-3 \
        --supcon-weight 0.1 \
        --supcon-temperature 0.2 \
        --physical-weight 0.10

    mkdir -p "${SAVE_DIR}"
    "${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
        --dataset_profile lora25 \
        --dataset_path "${LORA_NPZ}" \
        --save_dir "${SAVE_DIR}" \
        --checkpoint "${CHECKPOINT}" \
        --closedset_backbone lora_chirp \
        --seed "${seed}" --initial_known_classes 10 --round_size 5 --num_rounds 3 \
        --use_supcon --supcon_weight 0.1 \
        --discovery_backend gpcc_recording_consensus \
        --recording_consensus_threshold 0.65 \
        --radcil_registration_top_fraction 0.80 \
        --radcil_registration_min_per_class 1 \
        --cross_day_repr_adaptation \
        --cross_day_repr_epochs 2 \
        --cross_day_repr_batch_size 128 \
        --cross_day_repr_lr 1e-5 \
        --cross_day_repr_consistency_weight 0.5 \
        --cross_day_repr_coral_weight 0.05 \
        --lora_ssl_adaptation \
        --lora_ssl_epochs 3 \
        --lora_ssl_batch_size 128 \
        --lora_ssl_lr 1e-5 \
        --lora_ssl_ce_weight 1.0 \
        --lora_ssl_instance_weight 0.5 \
        --lora_ssl_teacher_weight 0.2 \
        --lora_ssl_temperature 0.2 \
        --enable_joint_discovery_refinement \
        --enable_lora_recording_eval \
        --cil_head_warmup_epochs 2 \
        --cil_joint_epochs 8 \
        --cil_replay_weight 3.0 \
        --radcil_old_new_batch_ratio 2.0 \
        --memory_per_class 20
done

"${PYTHON_BIN}" tools/stage38_lora_physical_pretrain_multiseed_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --seeds "${SEEDS}" \
    --output "${ROOT}/STAGE38_LORA_PHYSICAL_PRETRAIN_MULTISEED_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage38_lora_physical_pretrain_multiseed_summary_${JOB_ID}.json"

echo "Stage 38 LoRa 物理域自监督预训练三种子确认完成。"
