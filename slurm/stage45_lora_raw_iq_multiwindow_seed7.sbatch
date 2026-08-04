#!/bin/bash
# Stage 45：LoRa Setup 1 完整原始 I/Q 长窗/多窗 seed7 验证。
#
# 目的：
#   1. 不再依赖 9.7MB compact 子集或远程 range 下载；
#   2. 直接从 Stage44 已下载的完整 IQ_*.dat 读取 strict 协议需要的短片段；
#   3. 对比 1024 点 aligned symbol 下 14/28 个 symbol per transmission 是否
#      能把 Stage27 Chirp 候选的 LoRa R3 Overall 继续往上推。
#
# 使用方式：
#   sbatch slurm/stage45_lora_raw_iq_multiwindow_seed7.sbatch
#   如需覆盖原始目录：
#   RAW_DIR=/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage44/raw_setup1_iq \
#     sbatch slurm/stage45_lora_raw_iq_multiwindow_seed7.sbatch
#
# 协议边界：
#   raw_dir 只提供原始波形来源；strict split、标签顺序和 held-out eval 边界仍由
#   现有 LoRa loader/manifest 决定，不使用未知真值做聚类或调参。
#SBATCH --job-name=stage45_lora_raw_multi
#SBATCH --partition=gpu
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage45/slurm_lora_raw_iq_multiwindow_seed7_%j.out
#SBATCH --error=results/stage45/slurm_lora_raw_iq_multiwindow_seed7_%j.err

set -euo pipefail

module load slurm

DATA_ROOT="${DATA_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
ROOT="results/stage45"
RAW_DIR="${RAW_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage44/raw_setup1_iq}"
CACHE_DIR="${LORA25_CACHE_DIR:-${ROOT}/cache_raw_setup1_${JOB_ID}}"
RAW_PROBE_NPZ="${ROOT}/lora25_raw_probe_multiwindow_${JOB_ID}.npz"
SOURCE_MANIFEST="${ROOT}/lora25_source_manifest_raw_multiwindow_${JOB_ID}.json"

mkdir -p "${ROOT}" "${CACHE_DIR}"

echo "Stage45 raw_dir=${RAW_DIR}"
echo "Stage45 cache_dir=${CACHE_DIR}"

# 先把完整 IQ_*.dat 中需要的一小段截成 cache。这里按 s28 的最大窗口需求
# 生成 range，后续 s14/s28 aligned 数据集都复用同一个 cache，避免重复读原始包。
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

run_variant() {
    local variant="$1"
    local symbols="$2"
    local npz_path="${ROOT}/lora25_diffdays_indoor_raw1024_${variant}_${JOB_ID}.npz"
    local align_manifest="${ROOT}/lora25_alignment_manifest_raw1024_${variant}_${JOB_ID}.json"
    local save_dir="${ROOT}/lora_raw_${variant}_seed7_${JOB_ID}"
    local checkpoint="${save_dir}/closedset_lora25_lora_chirp_10known.pth"

    mkdir -p "${save_dir}"

    if [ ! -f "${npz_path}" ]; then
        "${PYTHON_BIN}" datasets/lora25_compact/tools/build_lora25_aligned.py \
            --cache_dir "${CACHE_DIR}" \
            --output "${npz_path}" \
            --alignment_manifest "${align_manifest}" \
            --symbols_per_transmission "${symbols}" \
            --decimation 1 \
            --representation raw
    fi

    "${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
        --dataset_profile lora25 \
        --dataset_path "${npz_path}" \
        --save_dir "${save_dir}" \
        --checkpoint "${checkpoint}" \
        --train_closedset --epochs 20 \
        --closedset_backbone lora_chirp \
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
}

run_variant "s14" 14
run_variant "s28" 28

"${PYTHON_BIN}" tools/stage45_lora_raw_iq_multiwindow_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --variants s14 s28 \
    --output "${ROOT}/STAGE45_LORA_RAW_IQ_MULTIWINDOW_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage45_lora_raw_iq_multiwindow_seed7_summary_${JOB_ID}.json"
