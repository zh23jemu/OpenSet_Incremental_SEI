#!/bin/bash
# Stage 33：LoRa recording-consensus discovery-only seed7 矩阵。
#
# 先训练一次固定的 LoRa Chirp closed-set backbone，再复用同一 checkpoint
# 比较普通 GPCC 和三个 recording 一致性阈值。所有任务都停在 discovery-only，
# 不会把未验证的伪标签带入增量后端。

#SBATCH --job-name=osei-s33-lora-disc
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage33/slurm_lora_recording_consensus_seed7_%j.out
#SBATCH --error=results/stage33/slurm_lora_recording_consensus_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-${DATA_ROOT}/datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
ROOT="results/stage33"
CHECKPOINT="${ROOT}/lora_chirp_closedset_seed7_${JOB_ID}.pth"

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}"

# 计划文件是实验审计材料，不参与训练逻辑。
"${PYTHON_BIN}" tools/stage33_lora_recording_consensus_plan.py \
    --output "${ROOT}/stage33_lora_recording_consensus_plan.json"

# 只训练一次初始 backbone，后续四个变体共享该 checkpoint。
"${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
    --dataset_profile lora25 \
    --dataset_path "${LORA_NPZ}" \
    --save_dir "${ROOT}/lora_recording_consensus_pretrain_seed7_${JOB_ID}" \
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
    --discovery_only \
    --disable_cil_baselines --disable_visualization

for VARIANT in gpcc consensus_t0p65 consensus_t0p75 consensus_t0p85; do
    SAVE_DIR="${ROOT}/lora_recording_consensus_${VARIANT}_seed7_${JOB_ID}"
    mkdir -p "${SAVE_DIR}"
    BACKEND="gpcc"
    EXTRA_ARGS=()
    if [[ "${VARIANT}" != "gpcc" ]]; then
        BACKEND="gpcc_recording_consensus"
        case "${VARIANT}" in
            consensus_t0p65) EXTRA_ARGS+=(--recording_consensus_threshold 0.65) ;;
            consensus_t0p75) EXTRA_ARGS+=(--recording_consensus_threshold 0.75) ;;
            consensus_t0p85) EXTRA_ARGS+=(--recording_consensus_threshold 0.85) ;;
        esac
    fi
    "${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
        --dataset_profile lora25 \
        --dataset_path "${LORA_NPZ}" \
        --save_dir "${SAVE_DIR}" \
        --checkpoint "${CHECKPOINT}" \
        --closedset_backbone lora_chirp \
        --seed 7 --initial_known_classes 10 --round_size 5 --num_rounds 3 \
        --use_supcon --supcon_weight 0.1 \
        --discovery_backend "${BACKEND}" \
        "${EXTRA_ARGS[@]}" \
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
        --discovery_only \
        --disable_cil_baselines --disable_visualization
done

"${PYTHON_BIN}" tools/stage33_lora_recording_consensus_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --output "${ROOT}/STAGE33_LORA_RECORDING_CONSENSUS_DISCOVERY_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage33_lora_recording_consensus_summary_${JOB_ID}.json"

echo "Stage 33 LoRa recording-consensus discovery-only seed7 完成。"
