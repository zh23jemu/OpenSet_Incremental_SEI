#!/bin/bash
# Stage 30：LoRa Chirp + 高阈值旧类原型路由 seed7 验证。
#
# Stage 29 证明旧类原型路由能提升 Old，但 0.45-0.85 阈值网格会严重压低
# New。本实验不改训练结构，只把路由阈值候选提高到 0.90-0.99，
# 验证是否存在“少量救旧类、不压塌新类”的保守折中。

#SBATCH --job-name=osei-s30-lora
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage30/slurm_lora_old_route_cons_seed7_%j.out
#SBATCH --error=results/stage30/slurm_lora_old_route_cons_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
SAVE_DIR="results/stage30/lora_old_prototype_route_seed7_${JOB_ID}"
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
    --radcil_old_prototype_route_candidates "0.90,0.95,0.98,0.99"

"${PYTHON_BIN}" tools/stage29_lora_old_prototype_route_report.py \
    --root results/stage30 \
    --job-id "${JOB_ID}" \
    --stage-label "Stage 30 LoRa 保守旧类原型路由 seed7" \
    --output "results/stage30/STAGE30_LORA_OLD_PROTOTYPE_ROUTE_CONSERVATIVE_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage30/stage30_lora_old_prototype_route_conservative_seed7_summary_${JOB_ID}.json"

echo "Stage 30 LoRa conservative old-prototype route seed7 验证完成。"
