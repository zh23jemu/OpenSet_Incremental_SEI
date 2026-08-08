#!/bin/bash
# Stage82：LoRa transmission-symbol pooling discovery-only seed7。
#
# 先只验证 transmission pooling 是否改善严格 symbol-level discovery。
# 只有报告通过门槛后，才提交对应的完整 CIL 脚本，避免把无效前端带入长训练。
# 输入仍为 Setup 1 raw-s28 strict NPZ，不复制原始 58G 数据。

#SBATCH --job-name=stage82_lora_tx_pool_disc
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage82/slurm_lora_tx_pool_discovery_seed7_%j.out
#SBATCH --error=results/stage82/slurm_lora_tx_pool_discovery_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
ROOT="results/stage82"
LORA_NPZ="${LORA25_NPZ:-${PROJECT_ROOT}/results/stage46/lora25_diffdays_indoor_raw1024_s28_46103936.npz}"
RAW_DIR="${LORA25_RAW_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage44/raw_setup1_iq}"
CACHE_DIR="${LORA25_CACHE_DIR:-${ROOT}/cache_raw_setup1_s28_${JOB_ID}}"
RAW_PROBE_NPZ="${ROOT}/lora25_raw_probe_s28_${JOB_ID}.npz"
SOURCE_MANIFEST="${ROOT}/lora25_source_manifest_raw_s28_${JOB_ID}.json"
ALIGN_MANIFEST="${ROOT}/lora25_alignment_manifest_raw1024_s28_${JOB_ID}.json"
SAVE_DIR="${ROOT}/lora_tx_pool_discovery_seed7_${JOB_ID}"

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}" "${SAVE_DIR}" "${CACHE_DIR}"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 127
fi
if [ ! -f "${LORA_NPZ}" ]; then
    echo "Stage82 raw-s28 NPZ missing, rebuilding from raw_dir=${RAW_DIR}"
    LORA_NPZ="${ROOT}/lora25_diffdays_indoor_raw1024_s28_${JOB_ID}.npz"
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
    if [ ! -f "${LORA_NPZ}" ]; then
        "${PYTHON_BIN}" datasets/lora25_compact/tools/build_lora25_aligned.py \
            --cache_dir "${CACHE_DIR}" \
            --output "${LORA_NPZ}" \
            --alignment_manifest "${ALIGN_MANIFEST}" \
            --symbols_per_transmission 28 \
            --decimation 1 \
            --representation raw
    fi
fi

"${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
    --dataset_profile lora25 \
    --dataset_path "${LORA_NPZ}" \
    --save_dir "${SAVE_DIR}" \
    --closedset_backbone lora_chirp \
    --seed 7 --initial_known_classes 10 --round_size 5 --num_rounds 3 \
    --use_supcon --supcon_weight 0.1 \
    --disable_cil_baselines --disable_visualization \
    --discovery_backend gpcc \
    --lora_transmission_pooling \
    --discovery_only

"${PYTHON_BIN}" tools/stage82_lora_transmission_pooling_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --mode discovery \
    --output "${ROOT}/STAGE82_LORA_TRANSMISSION_POOLING_DISCOVERY_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage82_lora_transmission_pooling_discovery_seed7_summary_${JOB_ID}.json"

echo "Stage82 LoRa transmission pooling discovery-only seed7 完成。"
