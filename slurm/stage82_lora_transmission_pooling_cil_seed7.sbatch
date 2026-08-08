#!/bin/bash
# Stage82：LoRa transmission-symbol pooling 完整 CIL seed7。
#
# 只有 discovery-only 报告达到门槛后才提交本脚本。正式成绩仍以
# symbol-level incremental_results.csv 为主，recording-level 仅作诊断。

#SBATCH --job-name=stage82_lora_tx_pool_cil
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage82/slurm_lora_tx_pool_cil_seed7_%j.out
#SBATCH --error=results/stage82/slurm_lora_tx_pool_cil_seed7_%j.err

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
SAVE_DIR="${ROOT}/lora_tx_pool_cil_seed7_${JOB_ID}"

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
    --discovery_backend gpcc_recording_consensus \
    --recording_consensus_threshold 0.65 \
    --lora_transmission_pooling \
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
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0 \
    --memory_per_class 20

"${PYTHON_BIN}" tools/stage82_lora_transmission_pooling_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --mode cil \
    --output "${ROOT}/STAGE82_LORA_TRANSMISSION_POOLING_CIL_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage82_lora_transmission_pooling_cil_seed7_summary_${JOB_ID}.json"

echo "Stage82 LoRa transmission pooling CIL seed7 完成。"
