#!/bin/bash
# Stage 76：LoRa 旧类跨天校准与当前伪新类联合 CIL seed7。
#
# 该实验把 Day2/Day3/Day4 的旧设备标注 IQ_1-7 直接放进每轮 CIL
# 优化步，与当前轮伪新类和 replay 旧类共同训练。它不读取 IQ_8-10，
# 也不使用旧类跨天数据做 held-out 评估或超参数选择。
#
# 与 Stage75 的区别：Stage75 只在 discovery 前做一次跨天表征适配；
# Stage76 把监督信号延伸到 head warmup、joint backbone 和第二次
# discovery refinement 的 CIL 训练中，用来验证“训练后又漂回去”这一
# 结构性风险能否被直接抑制。

#SBATCH --job-name=stage76-lora-joint-cil
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage76/slurm_lora_old_day_joint_cil_seed7_%j.out
#SBATCH --error=results/stage76/slurm_lora_old_day_joint_cil_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_ROOT}"
DATA_ROOT="${DATA_ROOT:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python"
fi
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 127
fi

JOB_ID="${SLURM_JOB_ID:-manual}"
ROOT="results/stage76"
SAVE_DIR="${ROOT}/lora_old_day_joint_cil_seed7_${JOB_ID}"
CHECKPOINT="${SAVE_DIR}/closedset_lora25_lora_chirp_10known.pth"
RAW_DIR="${LORA_RAW_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage44/raw_setup1_iq}"
CACHE_DIR="${ROOT}/cache_raw_setup1_s28_${JOB_ID}"
RAW_PROBE_NPZ="${ROOT}/lora25_raw_probe_s28_${JOB_ID}.npz"
LORA_NPZ="${LORA25_NPZ:-${ROOT}/lora25_diffdays_indoor_raw1024_s28_${JOB_ID}.npz}"
ALIGN_MANIFEST="${ROOT}/lora25_alignment_manifest_raw1024_s28_${JOB_ID}.json"

mkdir -p "${SAVE_DIR}" "${CACHE_DIR}"
echo "Stage76 project=${PROJECT_ROOT}"
echo "Stage76 data_root=${DATA_ROOT}"
echo "Stage76 lora_npz=${LORA_NPZ}"
echo "Stage76 raw_dir=${RAW_DIR}"
echo "Stage76 python=${PYTHON_BIN}"

if [ ! -f "${LORA_NPZ}" ]; then
    echo "Stage76 input NPZ missing; rebuild the same raw-IQ s28 input before training."
    "${PYTHON_BIN}" datasets/lora25_compact/tools/build_lora25_compact.py \
        --output "${RAW_PROBE_NPZ}" \
        --cache_dir "${CACHE_DIR}" \
        --manifest "${ROOT}/lora25_source_manifest_raw_s28_${JOB_ID}.json" \
        --raw_dir "${RAW_DIR}" \
        --window_len 1024 \
        --stride 1024 \
        --windows_per_transmission 30
    "${PYTHON_BIN}" datasets/lora25_compact/tools/build_lora25_aligned.py \
        --cache_dir "${CACHE_DIR}" \
        --output "${LORA_NPZ}" \
        --alignment_manifest "${ALIGN_MANIFEST}" \
        --symbols_per_transmission 28 \
        --decimation 1 \
        --representation raw
fi

"${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
    --dataset_profile lora25 \
    --dataset_path "${LORA_NPZ}" \
    --save_dir "${SAVE_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --train_closedset --epochs 20 \
    --closedset_backbone lora_chirp \
    --seed 7 --initial_known_classes 10 --round_size 5 --num_rounds 3 \
    --use_supcon --supcon_weight 0.1 \
    --disable_cil_baselines --disable_visualization \
    --discovery_backend gpcc_recording_consensus \
    --recording_consensus_threshold 0.65 \
    --lora_ssl_adaptation \
    --lora_ssl_epochs 3 \
    --lora_ssl_batch_size 96 \
    --lora_ssl_lr 1e-5 \
    --lora_ssl_ce_weight 1.0 \
    --lora_ssl_instance_weight 0.5 \
    --lora_ssl_teacher_weight 0.2 \
    --lora_ssl_temperature 0.2 \
    --lora_old_day_joint_cil \
    --lora_old_day_calibration_raw_dir "${RAW_DIR}" \
    --lora_old_day_calibration_transmissions 1,2,3,4,5,6,7 \
    --lora_old_day_calibration_symbols 28 \
    --lora_old_day_calibration_decimation 1 \
    --lora_old_day_calibration_representation raw \
    --lora_old_day_joint_cil_batch_size 96 \
    --lora_old_day_joint_cil_ce_weight 1.0 \
    --lora_old_day_joint_cil_feature_weight 0.5 \
    --enable_joint_discovery_refinement \
    --enable_lora_recording_eval \
    --cil_head_warmup_epochs 2 \
    --cil_joint_epochs 8 \
    --incremental_batch_size 96 \
    --test_batch_size 192 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0

echo "Stage76 LoRa 旧类跨天校准与联合 CIL seed7 完成。"
