#!/bin/bash
# Stage 63：LoRa raw s28 增强一致性物理预训练 seed7。
#
# 背景：
#   Stage61/62 说明 LoRa 低分不是单纯聚类或现有后端小机制能解决。
#   本阶段不再调 old-route、DOI fusion、阈值或注册比例，而是增强初始
#   LoRa Chirp backbone 的训练目标。
#
# 验证内容：
#   1. 只用 Day1 IQ_1-6 train / IQ_7 validation 训练 checkpoint。
#   2. 在 CE + SupCon + 物理描述符回归外，加入同一样本两种增强视图的
#      logit/feature 一致性约束，提升跨增强稳定性。
#   3. 后续 CIL 固定 Stage48 raw s28 + recording-consensus 0.65 主配置。
#   4. seed7 不过门槛则不扩三种子。
#SBATCH --job-name=stage63_lora_s28_consistency
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage63/slurm_lora_s28_consistency_seed7_%j.out
#SBATCH --error=results/stage63/slurm_lora_s28_consistency_seed7_%j.err

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
ROOT="results/stage63"
S28_NPZ="${LORA25_S28_NPZ:-results/stage46/lora25_diffdays_indoor_raw1024_s28_46103936.npz}"
CHECKPOINT="${ROOT}/lora_s28_consistency_pretrain_checkpoint_seed7_${JOB_ID}.pth"
SUMMARY_CHECKPOINT="${ROOT}/stage63_lora_s28_consistency_checkpoint_summary_${JOB_ID}.json"
SAVE_DIR="${ROOT}/lora_s28_consistency_seed7_${JOB_ID}"
mkdir -p "${ROOT}" "${SAVE_DIR}"

echo "Stage63 data_root=${DATA_ROOT}"
echo "Stage63 s28_npz=${S28_NPZ}"
echo "Stage63 checkpoint=${CHECKPOINT}"
echo "Stage63 python=${PYTHON_BIN}"
if [ ! -f "${S28_NPZ}" ]; then
    echo "Missing s28 NPZ: ${S28_NPZ}. Run Stage46 first or pass LORA25_S28_NPZ." >&2
    exit 2
fi

"${PYTHON_BIN}" tools/stage38_lora_physical_pretrain.py \
    --dataset-path "${S28_NPZ}" \
    --output "${CHECKPOINT}" \
    --summary-json "${SUMMARY_CHECKPOINT}" \
    --seed 7 \
    --epochs 30 \
    --batch-size 96 \
    --lr 8e-4 \
    --supcon-weight 0.1 \
    --physical-weight 0.05 \
    --view-logit-consistency-weight 0.30 \
    --view-feature-consistency-weight 0.10 \
    --view-consistency-temperature 2.0

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

"${PYTHON_BIN}" tools/stage63_lora_s28_consistency_pretrain_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --output "${ROOT}/STAGE63_LORA_S28_CONSISTENCY_PRETRAIN_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage63_lora_s28_consistency_pretrain_summary_${JOB_ID}.json"
