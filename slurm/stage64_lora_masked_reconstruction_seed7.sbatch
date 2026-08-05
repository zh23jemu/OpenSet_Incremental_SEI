#!/bin/bash
# Stage 64：LoRa raw s28 masked reconstruction 预训练 seed7。
#
# 这是 Stage63 后的更大结构候选：不调后端小参数，而是先用原始 I/Q
# 遮挡重建任务预训练 backbone，再接 Stage48 raw s28 + rec065 主流程。
# 预训练只使用 Day1 train 和协议允许的 Day2-4 discovery/enrollment 未标注样本；
# 不读取未知类真值，不访问 held-out eval。
#SBATCH --job-name=stage64_lora_masked_recon
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage64/slurm_lora_masked_recon_seed7_%j.out
#SBATCH --error=results/stage64/slurm_lora_masked_recon_seed7_%j.err

set -euo pipefail

module load slurm

DATA_ROOT="${DATA_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python"
fi
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 127
fi

JOB_ID="${SLURM_JOB_ID:-manual}"
ROOT="results/stage64"
S28_NPZ="${LORA25_S28_NPZ:-results/stage46/lora25_diffdays_indoor_raw1024_s28_46103936.npz}"
CHECKPOINT="${ROOT}/lora_s28_masked_reconstruction_checkpoint_seed7_${JOB_ID}.pth"
SUMMARY_CHECKPOINT="${ROOT}/stage64_lora_masked_reconstruction_checkpoint_summary_${JOB_ID}.json"
SAVE_DIR="${ROOT}/lora_s28_masked_recon_seed7_${JOB_ID}"
mkdir -p "${ROOT}" "${SAVE_DIR}"

echo "Stage64 data_root=${DATA_ROOT}"
echo "Stage64 s28_npz=${S28_NPZ}"
echo "Stage64 checkpoint=${CHECKPOINT}"
echo "Stage64 python=${PYTHON_BIN}"
if [ ! -f "${S28_NPZ}" ]; then
    echo "Missing s28 NPZ: ${S28_NPZ}. Run Stage46 first or pass LORA25_S28_NPZ." >&2
    exit 2
fi

"${PYTHON_BIN}" tools/stage64_lora_masked_reconstruction_pretrain.py \
    --dataset-path "${S28_NPZ}" \
    --output "${CHECKPOINT}" \
    --summary-json "${SUMMARY_CHECKPOINT}" \
    --seed 7 \
    --pretrain-epochs 20 \
    --finetune-epochs 20 \
    --batch-size 96 \
    --pretrain-lr 8e-4 \
    --finetune-lr 8e-4 \
    --mask-ratio 0.35 \
    --feature-consistency-weight 0.05 \
    --include-discovery-unlabeled

"${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
    --dataset_profile lora25 \
    --dataset_path "${S28_NPZ}" \
    --save_dir "${SAVE_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --closedset_backbone lora_chirp \
    --seed 7 --initial_known_classes 10 --round_size 5 --num_rounds 3 \
    --use_supcon --supcon_weight 0.1 \
    --disable_cil_baselines --disable_visualization \
    --discovery_backend gpcc_recording_consensus \
    --recording_consensus_threshold 0.65 \
    --cross_day_repr_adaptation \
    --cross_day_repr_epochs 2 \
    --cross_day_repr_batch_size 96 \
    --cross_day_repr_lr 1e-5 \
    --cross_day_repr_consistency_weight 0.5 \
    --cross_day_repr_coral_weight 0.05 \
    --lora_ssl_adaptation \
    --lora_ssl_epochs 3 \
    --lora_ssl_batch_size 96 \
    --lora_ssl_lr 1e-5 \
    --lora_ssl_ce_weight 1.0 \
    --lora_ssl_instance_weight 0.5 \
    --lora_ssl_teacher_weight 0.2 \
    --lora_ssl_temperature 0.2 \
    --enable_joint_discovery_refinement \
    --enable_lora_recording_eval \
    --cil_head_warmup_epochs 2 \
    --cil_joint_epochs 8 \
    --incremental_batch_size 96 \
    --test_batch_size 192 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0

"${PYTHON_BIN}" tools/stage64_lora_masked_reconstruction_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --output "${ROOT}/STAGE64_LORA_MASKED_RECONSTRUCTION_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage64_lora_masked_reconstruction_summary_${JOB_ID}.json"
