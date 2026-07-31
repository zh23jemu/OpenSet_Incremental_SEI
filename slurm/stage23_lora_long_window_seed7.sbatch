#!/bin/bash
# Stage 23：LoRa 长窗 aligned 子集 seed7 验证。
#
# 目的：
#   当前 256 点 LoRa 子集在三种子上仍低分。该任务不改变 strict 类别划分，
#   只把 aligned symbol 从 decimation=4 的 256 点改成 decimation=1 的
#   1024 点，验证更完整 chirp 形状是否能同时改善聚类和增量识别。
#
# 数据边界：
#   只下载/重建当前实验需要的 Different Days Indoor Setup 记录；
#   Day1 IQ_1-6 训练，IQ_7 验证，IQ_8-10 仍只用于 held-out evaluation。

#SBATCH --job-name=osei-s23-lora-long
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage23/slurm_lora_long_window_seed7_%j.out
#SBATCH --error=results/stage23/slurm_lora_long_window_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"

cd "${PROJECT_ROOT}"
mkdir -p results/stage23

CACHE_DIR="${LORA25_CACHE_DIR:-${DATA_ROOT}/datasets/lora25_compact/cache_stage23_long}"
RAW_PROBE_NPZ="results/stage23/lora25_raw_probe_${JOB_ID}.npz"
SOURCE_MANIFEST="results/stage23/lora25_source_manifest_long_${JOB_ID}.json"
LONG_NPZ="${LORA25_LONG_NPZ:-results/stage23/lora25_diffdays_indoor_aligned_group_1024_${JOB_ID}.npz}"
ALIGN_MANIFEST="results/stage23/lora25_alignment_group_manifest_1024_${JOB_ID}.json"
SAVE_DIR="results/stage23/lora_long1024_joint_seed7_${JOB_ID}"
CHECKPOINT="${SAVE_DIR}/closedset_lora25_long1024_seed7.pth"

echo "Stage 23 使用 cache_dir=${CACHE_DIR}"
mkdir -p "${CACHE_DIR}" "${SAVE_DIR}"

# build_lora25_compact.py 负责按 strict manifest 下载必要 range 并填充 cache。
# 这里的 RAW_PROBE_NPZ 只是为了复用其下载逻辑，后续训练使用 aligned LONG_NPZ。
if [ ! -f "${LONG_NPZ}" ]; then
    "${PYTHON_BIN}" datasets/lora25_compact/tools/build_lora25_compact.py \
        --output "${RAW_PROBE_NPZ}" \
        --cache_dir "${CACHE_DIR}" \
        --manifest "${SOURCE_MANIFEST}" \
        --workers 8 \
        --window_len 1024 \
        --stride 1024 \
        --windows_per_transmission 16

    "${PYTHON_BIN}" datasets/lora25_compact/tools/build_lora25_aligned.py \
        --cache_dir "${CACHE_DIR}" \
        --output "${LONG_NPZ}" \
        --alignment_manifest "${ALIGN_MANIFEST}" \
        --symbols_per_transmission 14 \
        --decimation 1 \
        --representation raw
fi

"${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
    --dataset_profile lora25 \
    --dataset_path "${LONG_NPZ}" \
    --save_dir "${SAVE_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --train_closedset \
    --epochs 20 \
    --seed 7 --initial_known_classes 10 --round_size 5 --num_rounds 3 \
    --use_supcon --supcon_weight 0.1 \
    --disable_cil_baselines --disable_visualization \
    --discovery_backend gpcc \
    --cross_day_repr_adaptation \
    --cross_day_repr_epochs 2 \
    --cross_day_repr_batch_size 96 \
    --cross_day_repr_lr 1e-5 \
    --cross_day_repr_consistency_weight 0.5 \
    --cross_day_repr_coral_weight 0.05 \
    --enable_joint_discovery_refinement \
    --cil_head_warmup_epochs 2 \
    --cil_joint_epochs 8 \
    --incremental_batch_size 96 \
    --test_batch_size 192 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0

"${PYTHON_BIN}" tools/stage23_lora_long_window_report.py \
    --save-dir "${SAVE_DIR}" \
    --job-id "${JOB_ID}" \
    --sample-length 1024 \
    --output "results/stage23/STAGE23_LORA_LONG_WINDOW_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage23/stage23_lora_long_window_seed7_summary_${JOB_ID}.json"

echo "Stage 23 LoRa 长窗 seed7 完成。"
