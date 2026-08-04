#!/bin/bash
# Stage 47：LoRa s28 seed31 新类塌缩修复小矩阵。
#
# 背景：
#   Stage46 s28 三种子提升了 Overall/Old/遗忘，但 seed31 R3 New Acc
#   从 Stage45 seed7 的 0.5571 掉到 0.2952。这里不继续扩大窗口数，而是
#   针对伪标签注册与 recording 一致性做最小矩阵，先只跑 seed31。
#SBATCH --job-name=stage47_lora_s28_rescue
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage47/slurm_lora_s28_new_class_rescue_seed31_%j.out
#SBATCH --error=results/stage47/slurm_lora_s28_new_class_rescue_seed31_%j.err

set -euo pipefail

module load slurm

DATA_ROOT="${DATA_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    # 干净 worktree 没有 .venv 时，回退到主项目已经验证过的虚拟环境。
    PYTHON_BIN="/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python"
fi
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 127
fi

JOB_ID="${SLURM_JOB_ID:-manual}"
ROOT="results/stage47"
RAW_DIR="${RAW_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage44/raw_setup1_iq}"
CACHE_DIR="${LORA25_CACHE_DIR:-${ROOT}/cache_raw_setup1_s28_${JOB_ID}}"
RAW_PROBE_NPZ="${ROOT}/lora25_raw_probe_s28_${JOB_ID}.npz"
SOURCE_MANIFEST="${ROOT}/lora25_source_manifest_raw_s28_${JOB_ID}.json"
S28_NPZ="${LORA25_S28_NPZ:-results/stage46/lora25_diffdays_indoor_raw1024_s28_46103936.npz}"
ALIGN_MANIFEST="${ROOT}/lora25_alignment_manifest_raw1024_s28_${JOB_ID}.json"

mkdir -p "${ROOT}" "${CACHE_DIR}"

echo "Stage47 raw_dir=${RAW_DIR}"
echo "Stage47 s28_npz=${S28_NPZ}"
echo "Stage47 python=${PYTHON_BIN}"

# 优先复用 Stage46 的 s28 NPZ；若该文件不存在，则从完整原始 I/Q 重建。
if [ ! -f "${S28_NPZ}" ]; then
    S28_NPZ="${ROOT}/lora25_diffdays_indoor_raw1024_s28_${JOB_ID}.npz"
    "${PYTHON_BIN}" datasets/lora25_compact/tools/build_lora25_compact.py \
        --output "${RAW_PROBE_NPZ}" \
        --cache_dir "${CACHE_DIR}" \
        --manifest "${SOURCE_MANIFEST}" \
        --raw_dir "${RAW_DIR}" \
        --window_len 1024 \
        --stride 1024 \
        --windows_per_transmission 30
    "${PYTHON_BIN}" datasets/lora25_compact/tools/build_lora25_aligned.py \
        --cache_dir "${CACHE_DIR}" \
        --output "${S28_NPZ}" \
        --alignment_manifest "${ALIGN_MANIFEST}" \
        --symbols_per_transmission 28 \
        --decimation 1 \
        --representation raw
fi

run_variant() {
    local variant="$1"
    shift
    local save_dir="${ROOT}/lora_s28_seed31_${variant}_${JOB_ID}"
    local checkpoint="${save_dir}/closedset_lora25_lora_chirp_10known.pth"
    mkdir -p "${save_dir}"

    "${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
        --dataset_profile lora25 \
        --dataset_path "${S28_NPZ}" \
        --save_dir "${save_dir}" \
        --checkpoint "${checkpoint}" \
        --train_closedset --epochs 20 \
        --closedset_backbone lora_chirp \
        --seed 31 --initial_known_classes 10 --round_size 5 --num_rounds 3 \
        --use_supcon --supcon_weight 0.1 \
        --disable_cil_baselines --disable_visualization \
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
        --radcil_old_new_batch_ratio 2.0 \
        "$@"
}

run_variant "baseline" --discovery_backend gpcc
run_variant "rec065" --discovery_backend gpcc_recording_consensus --recording_consensus_threshold 0.65
run_variant "rec075" --discovery_backend gpcc_recording_consensus --recording_consensus_threshold 0.75
run_variant "top080" --discovery_backend gpcc --radcil_registration_top_fraction 0.80 --radcil_registration_min_per_class 1
run_variant "rec065_top080" --discovery_backend gpcc_recording_consensus --recording_consensus_threshold 0.65 --radcil_registration_top_fraction 0.80 --radcil_registration_min_per_class 1

"${PYTHON_BIN}" tools/stage47_lora_s28_new_class_rescue_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --variants baseline rec065 rec075 top080 rec065_top080 \
    --output "${ROOT}/STAGE47_LORA_S28_NEW_CLASS_RESCUE_SEED31_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage47_lora_s28_new_class_rescue_seed31_summary_${JOB_ID}.json"
