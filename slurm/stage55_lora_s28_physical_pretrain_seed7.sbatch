#!/bin/bash
# Stage 55：LoRa raw s28 物理域预训练 seed7。
#
# 背景：
#   Stage38 的物理描述符预训练曾在旧紧凑数据上有 seed7 正信号，但三种子不稳；
#   Stage48 已把 LoRa 正式候选切到完整 Setup 1 raw s28 + recording-consensus
#   0.65。这里不再调 head/ratio/recording loss，而是在当前最佳 raw s28
#   数据口径下重新验证物理域初始化是否能改善端到端 CIL。
#
# 验证内容：
#   1. 只用 Day1 IQ_1-6 / IQ_7 训练并选择 LoRa Chirp 物理描述符 checkpoint。
#   2. 固定 Stage48 raw s28 + rec065 + cross-day + LoRa SSL + joint CIL 主流程。
#   3. seed7 不过门槛则不扩三种子。
#
# 通过门槛：
#   相对 Stage48 seed7，R3 Overall 至少 +0.01；Old 不低于 Stage48 seed7；
#   New 下降不超过 0.02；Forgetting 恶化不超过 0.02。
#SBATCH --job-name=stage55_lora_s28_phys
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage55/slurm_lora_s28_physical_pretrain_seed7_%j.out
#SBATCH --error=results/stage55/slurm_lora_s28_physical_pretrain_seed7_%j.err

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
ROOT="results/stage55"
S28_NPZ="${LORA25_S28_NPZ:-results/stage46/lora25_diffdays_indoor_raw1024_s28_46103936.npz}"
CHECKPOINT="${ROOT}/lora_s28_physical_pretrain_checkpoint_seed7_${JOB_ID}.pth"
SUMMARY_CHECKPOINT="${ROOT}/stage55_lora_s28_physical_checkpoint_summary_${JOB_ID}.json"
SAVE_DIR="${ROOT}/lora_s28_physical_seed7_${JOB_ID}"
mkdir -p "${ROOT}" "${SAVE_DIR}"

echo "Stage55 data_root=${DATA_ROOT}"
echo "Stage55 s28_npz=${S28_NPZ}"
echo "Stage55 checkpoint=${CHECKPOINT}"
echo "Stage55 python=${PYTHON_BIN}"
if [ ! -f "${S28_NPZ}" ]; then
    echo "Missing s28 NPZ: ${S28_NPZ}. Run Stage46 first or pass LORA25_S28_NPZ." >&2
    exit 2
fi

"${PYTHON_BIN}" tools/stage38_lora_physical_pretrain.py \
    --dataset-path "${S28_NPZ}" \
    --output "${CHECKPOINT}" \
    --summary-json "${SUMMARY_CHECKPOINT}" \
    --seed 7 \
    --epochs 20 \
    --batch-size 96 \
    --lr 1e-3 \
    --supcon-weight 0.1 \
    --supcon-temperature 0.2 \
    --physical-weight 0.10

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

"${PYTHON_BIN}" tools/stage55_lora_s28_physical_pretrain_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --output "${ROOT}/STAGE55_LORA_S28_PHYSICAL_PRETRAIN_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage55_lora_s28_physical_pretrain_summary_${JOB_ID}.json"
