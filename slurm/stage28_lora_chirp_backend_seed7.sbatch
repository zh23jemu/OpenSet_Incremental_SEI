#!/bin/bash
# Stage 28：LoRa Chirp 表征下的后端隔离对照。
#
# 固定 Stage 27 的 Chirp、GPCC、cross-day、LoRa SSL 和联合 discovery-CIL；
# 这里只打开已有 DOI-style/iCaRL 等基线，使用同一批发现伪标签，
# 用于判断 R3 低分主要来自表征还是网络式 RADCIL 后端。

#SBATCH --job-name=osei-s28-lora
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage28/slurm_lora_chirp_backend_seed7_%j.out
#SBATCH --error=results/stage28/slurm_lora_chirp_backend_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
SAVE_DIR="results/stage28/lora_chirp_backend_seed7_${JOB_ID}"
CHECKPOINT="${SAVE_DIR}/closedset_lora25_lora_chirp_10known.pth"

mkdir -p "${SAVE_DIR}"

"${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
    --dataset_profile lora25 \
    --dataset_path "${LORA_NPZ}" \
    --save_dir "${SAVE_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --train_closedset --epochs 20 \
    --closedset_backbone lora_chirp \
    --seed 7 --initial_known_classes 10 --round_size 5 --num_rounds 3 \
    --use_supcon --supcon_weight 0.1 \
    --discovery_backend gpcc \
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

"${PYTHON_BIN}" tools/stage28_lora_chirp_backend_report.py \
    --root results/stage28 \
    --job-id "${JOB_ID}" \
    --output "results/stage28/STAGE28_LORA_CHIRP_BACKEND_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage28/stage28_lora_chirp_backend_seed7_summary_${JOB_ID}.json"

echo "Stage 28 LoRa Chirp backend seed7 对照完成。"
