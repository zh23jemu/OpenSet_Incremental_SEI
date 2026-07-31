#!/bin/bash
# Stage 29：LoRa Chirp + 保守旧类原型路由 seed7 验证。
#
# 该实验基于 Stage 28 的诊断：RADCIL 保新类但旧类弱，DOI-style 保旧类
# 但压低新类。这里只在网络旧类概率质量足够高时，用 replay 原型修正旧类
# 内部类别，避免全局把新类吸回旧类。

#SBATCH --job-name=osei-s29-lora
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage29/slurm_lora_old_route_seed7_%j.out
#SBATCH --error=results/stage29/slurm_lora_old_route_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
SAVE_DIR="results/stage29/lora_old_prototype_route_seed7_${JOB_ID}"
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
    --disable_cil_baselines --disable_visualization \
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
    --radcil_old_prototype_route \
    --radcil_old_prototype_route_candidates "0.45,0.55,0.65,0.75,0.85"

"${PYTHON_BIN}" tools/stage29_lora_old_prototype_route_report.py \
    --root results/stage29 \
    --job-id "${JOB_ID}" \
    --output "results/stage29/STAGE29_LORA_OLD_PROTOTYPE_ROUTE_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage29/stage29_lora_old_prototype_route_seed7_summary_${JOB_ID}.json"

echo "Stage 29 LoRa old-prototype route seed7 验证完成。"
