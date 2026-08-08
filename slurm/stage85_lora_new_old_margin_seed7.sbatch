#!/bin/bash
# Stage85：LoRa 当前轮新伪类 vs 旧类 logit margin seed7 验证。
#
# 背景：
#   Stage83/84 表明旧类跨天校准有信号，但后处理路由会牺牲 New。
#   本阶段不再做预测后处理，而是在 RADCIL 训练期保护当前轮新伪类：
#   让新伪类目标 logit 高于旧类最大 logit 一个 margin。
#
# 严格边界：
#   该损失只使用当前 discovery 伪标签、sample weight 和旧类输出边界；
#   不读取未知真值，不读取 IQ_8-10 held-out 标签做训练或选参。
#SBATCH --job-name=stage85-lora-new-margin
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python"
fi
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 127
fi

JOB_ID="${SLURM_JOB_ID:-manual}"
ROOT="results/stage85"
RUNTIME_ROOT="/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage85_runtime"
LOG_DIR="${RUNTIME_ROOT}/logs"
S28_NPZ="${LORA25_S28_NPZ:-results/stage46/lora25_diffdays_indoor_raw1024_s28_46103936.npz}"
SAVE_DIR="${ROOT}/lora_new_old_margin_seed7_${JOB_ID}"
CHECKPOINT="${SAVE_DIR}/closedset_lora25_lora_chirp_10known.pth"
MARGIN_WEIGHT="${MARGIN_WEIGHT:-0.25}"
MARGIN_VALUE="${MARGIN_VALUE:-0.5}"

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}" "${SAVE_DIR}" "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/stage85_lora_new_old_margin_seed7_${JOB_ID}.log"

{
    echo "Stage85 job=${JOB_ID}"
    echo "Stage85 s28_npz=${S28_NPZ}"
    echo "Stage85 save_dir=${SAVE_DIR}"
    echo "Stage85 margin_weight=${MARGIN_WEIGHT}"
    echo "Stage85 margin=${MARGIN_VALUE}"

    if [ ! -f "${S28_NPZ}" ]; then
        echo "Missing s28 NPZ: ${S28_NPZ}. Run Stage46 first or pass LORA25_S28_NPZ." >&2
        exit 2
    fi

    "${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
        --dataset_profile lora25 \
        --dataset_path "${S28_NPZ}" \
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
        --radcil_old_new_batch_ratio 2.0 \
        --radcil_new_old_margin_weight "${MARGIN_WEIGHT}" \
        --radcil_new_old_margin "${MARGIN_VALUE}"

    "${PYTHON_BIN}" tools/stage85_lora_new_old_margin_report.py \
        --save-dir "${SAVE_DIR}" \
        --job-id "${JOB_ID}" \
        --margin-weight "${MARGIN_WEIGHT}" \
        --margin "${MARGIN_VALUE}" \
        --output "${ROOT}/STAGE85_LORA_NEW_OLD_MARGIN_SEED7_REPORT_${JOB_ID}.md" \
        --summary-json "${ROOT}/stage85_lora_new_old_margin_summary_${JOB_ID}.json"
} >"${RUN_LOG}" 2>&1
