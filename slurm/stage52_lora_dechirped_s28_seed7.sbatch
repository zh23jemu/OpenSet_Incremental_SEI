#!/bin/bash
# Stage 52：LoRa dechirped s28 seed7 表示验证。
#
# 背景：
#   Stage48 raw s28 + recording-consensus 0.65 是当前 LoRa 正式候选；
#   Stage50/51 已证明 old:new ratio 和新类 head 吸收小机制不是突破口。
#   本阶段回到原始 I/Q 表示本身：对每个 1024 点 LoRa symbol 做 dechirp，
#   再移除主 FFT payload bin，保留频偏和模拟硬件残差。
#
# 通过门槛：
#   相对 Stage48 seed7，R3 Overall 至少 +0.01，Old 不下降，New 下降不超过
#   0.02，Forgetting 恶化不超过 0.02；不过门槛就不扩三种子。
#SBATCH --job-name=stage52_lora_dechirp_s7
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage52/slurm_lora_dechirped_s28_seed7_%j.out
#SBATCH --error=results/stage52/slurm_lora_dechirped_s28_seed7_%j.err

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
ROOT="results/stage52"
RAW_DIR="${RAW_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage44/raw_setup1_iq}"
CACHE_DIR="${LORA25_CACHE_DIR:-${ROOT}/cache_raw_setup1_dechirped_s28_${JOB_ID}}"
RAW_PROBE_NPZ="${ROOT}/lora25_raw_probe_dechirped_s28_${JOB_ID}.npz"
SOURCE_MANIFEST="${ROOT}/lora25_source_manifest_dechirped_s28_${JOB_ID}.json"
DECHIRP_NPZ="${LORA25_DECHIRPED_S28_NPZ:-${ROOT}/lora25_diffdays_indoor_dechirped_s28_${JOB_ID}.npz}"
ALIGN_MANIFEST="${ROOT}/lora25_alignment_manifest_dechirped_s28_${JOB_ID}.json"

mkdir -p "${ROOT}" "${CACHE_DIR}"

echo "Stage52 data_root=${DATA_ROOT}"
echo "Stage52 raw_dir=${RAW_DIR}"
echo "Stage52 cache_dir=${CACHE_DIR}"
echo "Stage52 dechirped_npz=${DECHIRP_NPZ}"
echo "Stage52 python=${PYTHON_BIN}"

if [ ! -d "${RAW_DIR}" ]; then
    echo "Missing RAW_DIR: ${RAW_DIR}" >&2
    exit 2
fi

if [ ! -f "${RAW_PROBE_NPZ}" ]; then
    "${PYTHON_BIN}" datasets/lora25_compact/tools/build_lora25_compact.py \
        --output "${RAW_PROBE_NPZ}" \
        --cache_dir "${CACHE_DIR}" \
        --manifest "${SOURCE_MANIFEST}" \
        --raw_dir "${RAW_DIR}" \
        --window_len 1024 \
        --stride 1024 \
        --windows_per_transmission 30
fi

if [ ! -f "${DECHIRP_NPZ}" ]; then
    "${PYTHON_BIN}" datasets/lora25_compact/tools/build_lora25_aligned.py \
        --cache_dir "${CACHE_DIR}" \
        --output "${DECHIRP_NPZ}" \
        --alignment_manifest "${ALIGN_MANIFEST}" \
        --symbols_per_transmission 28 \
        --decimation 1 \
        --representation dechirped
fi

SAVE_DIR="${ROOT}/lora_dechirped_s28_rec065_seed7_${JOB_ID}"
CHECKPOINT="${SAVE_DIR}/closedset_lora25_lora_chirp_10known.pth"
mkdir -p "${SAVE_DIR}"

"${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
    --dataset_profile lora25 \
    --dataset_path "${DECHIRP_NPZ}" \
    --save_dir "${SAVE_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --train_closedset --epochs 20 \
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

"${PYTHON_BIN}" tools/stage52_lora_dechirped_s28_seed7_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --output "${ROOT}/STAGE52_LORA_DECHIRPED_S28_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage52_lora_dechirped_s28_seed7_summary_${JOB_ID}.json"
