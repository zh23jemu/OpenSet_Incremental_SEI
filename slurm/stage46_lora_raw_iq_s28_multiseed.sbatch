#!/bin/bash
# Stage 46：LoRa Setup 1 原始 I/Q s28 三种子正式验证。
#
# 背景：
#   Stage45 seed7 显示，从完整 Setup 1 原始 I/Q 重切 1024 点、28 个
#   aligned symbol 的 s28 配置明显优于 Stage27 Chirp 候选；s14 未通过。
#
# 目的：
#   只扩展 s28 到 seed 7/13/31，确认单种子收益是否稳定。脚本不改变
#   LoRa strict split，只改变从原始 I/Q 切出的每个 transmission 样本数。
#SBATCH --job-name=stage46_lora_s28_ms
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage46/slurm_lora_raw_iq_s28_multiseed_%j.out
#SBATCH --error=results/stage46/slurm_lora_raw_iq_s28_multiseed_%j.err

set -euo pipefail

module load slurm

DATA_ROOT="${DATA_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
ROOT="results/stage46"
RAW_DIR="${RAW_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage44/raw_setup1_iq}"
CACHE_DIR="${LORA25_CACHE_DIR:-${ROOT}/cache_raw_setup1_s28_${JOB_ID}}"
RAW_PROBE_NPZ="${ROOT}/lora25_raw_probe_s28_${JOB_ID}.npz"
SOURCE_MANIFEST="${ROOT}/lora25_source_manifest_raw_s28_${JOB_ID}.json"
S28_NPZ="${LORA25_S28_NPZ:-${ROOT}/lora25_diffdays_indoor_raw1024_s28_${JOB_ID}.npz}"
ALIGN_MANIFEST="${ROOT}/lora25_alignment_manifest_raw1024_s28_${JOB_ID}.json"

mkdir -p "${ROOT}" "${CACHE_DIR}"

echo "Stage46 raw_dir=${RAW_DIR}"
echo "Stage46 cache_dir=${CACHE_DIR}"

# 生成 s28 所需的本地 range cache。这里从 Stage44 完整 IQ_*.dat 读取，不访问 OSU 网络。
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

# 生成 28-symbol aligned 数据。多个 seed 复用同一个 NPZ，保证只验证训练随机性。
if [ ! -f "${S28_NPZ}" ]; then
    "${PYTHON_BIN}" datasets/lora25_compact/tools/build_lora25_aligned.py \
        --cache_dir "${CACHE_DIR}" \
        --output "${S28_NPZ}" \
        --alignment_manifest "${ALIGN_MANIFEST}" \
        --symbols_per_transmission 28 \
        --decimation 1 \
        --representation raw
fi

run_seed() {
    local seed="$1"
    local save_dir="${ROOT}/lora_raw_s28_seed${seed}_${JOB_ID}"
    local checkpoint="${save_dir}/closedset_lora25_lora_chirp_10known.pth"
    mkdir -p "${save_dir}"

    "${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
        --dataset_profile lora25 \
        --dataset_path "${S28_NPZ}" \
        --save_dir "${save_dir}" \
        --checkpoint "${checkpoint}" \
        --train_closedset --epochs 20 \
        --closedset_backbone lora_chirp \
        --seed "${seed}" --initial_known_classes 10 --round_size 5 --num_rounds 3 \
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

run_seed 7
run_seed 13
run_seed 31

"${PYTHON_BIN}" tools/stage46_lora_raw_iq_s28_multiseed_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --seeds 7 13 31 \
    --output "${ROOT}/STAGE46_LORA_RAW_IQ_S28_MULTISEED_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage46_lora_raw_iq_s28_multiseed_summary_${JOB_ID}.json"
