#!/bin/bash
# Stage86：LoRa 旧类跨天监督 + 新类 margin 的平衡训练 seed7 验证。
#
# 背景：
#   Stage76 证明旧类跨天监督能抬 Old，但会明显压 New。
#   Stage85 证明新旧类 margin 能抬 New，但会明显伤 Old/Overall。
#   本阶段只做一个保守平衡候选：降低旧类跨天监督权重，同时加入较轻的新类
#   margin，让两种约束在训练期同时存在，验证是否能缓解单边优化的冲突。
#
# 严格边界：
#   旧类跨天监督只使用 Day2-4 旧设备 IQ_1-7，仍排除 IQ_8-10 held-out eval。
#   新类 margin 只使用当前 discovery 伪标签，不读取未知真值或 eval 真值。
#SBATCH --job-name=stage86-lora-balanced
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
ROOT="results/stage86"
RUNTIME_ROOT="/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage86_runtime"
LOG_DIR="${RUNTIME_ROOT}/logs"
S28_NPZ="${LORA25_S28_NPZ:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage45-worktree/results/stage46/lora25_diffdays_indoor_raw1024_s28_46103936.npz}"
RAW_DIR="${LORA_RAW_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage44/raw_setup1_iq}"
SAVE_DIR="${ROOT}/lora_balanced_old_new_training_seed7_${JOB_ID}"
CHECKPOINT="${SAVE_DIR}/closedset_lora25_lora_chirp_10known.pth"

# 这组默认值刻意保守：Stage76 的旧类监督从 1.0/0.5 降到 0.25/0.10；
# Stage85 的新类 margin 从 0.25 降到 0.10，避免把新旧冲突推向另一边。
OLD_DAY_CE_WEIGHT="${OLD_DAY_CE_WEIGHT:-0.25}"
OLD_DAY_FEATURE_WEIGHT="${OLD_DAY_FEATURE_WEIGHT:-0.10}"
MARGIN_WEIGHT="${MARGIN_WEIGHT:-0.10}"
MARGIN_VALUE="${MARGIN_VALUE:-0.5}"

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}" "${SAVE_DIR}" "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/stage86_lora_balanced_old_new_training_seed7_${JOB_ID}.log"

{
    echo "Stage86 job=${JOB_ID}"
    echo "Stage86 s28_npz=${S28_NPZ}"
    echo "Stage86 raw_dir=${RAW_DIR}"
    echo "Stage86 save_dir=${SAVE_DIR}"
    echo "Stage86 old_day_ce_weight=${OLD_DAY_CE_WEIGHT}"
    echo "Stage86 old_day_feature_weight=${OLD_DAY_FEATURE_WEIGHT}"
    echo "Stage86 margin_weight=${MARGIN_WEIGHT}"
    echo "Stage86 margin=${MARGIN_VALUE}"

    if [ ! -f "${S28_NPZ}" ]; then
        echo "Missing s28 NPZ: ${S28_NPZ}. Run Stage46 first or pass LORA25_S28_NPZ." >&2
        exit 2
    fi
    if [ ! -d "${RAW_DIR}" ]; then
        echo "Missing LoRa raw calibration dir: ${RAW_DIR}. Pass LORA_RAW_DIR if needed." >&2
        exit 3
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
        --lora_old_day_joint_cil \
        --lora_old_day_calibration_raw_dir "${RAW_DIR}" \
        --lora_old_day_calibration_transmissions 1,2,3,4,5,6,7 \
        --lora_old_day_calibration_symbols 28 \
        --lora_old_day_calibration_decimation 1 \
        --lora_old_day_calibration_representation raw \
        --lora_old_day_joint_cil_batch_size 96 \
        --lora_old_day_joint_cil_ce_weight "${OLD_DAY_CE_WEIGHT}" \
        --lora_old_day_joint_cil_feature_weight "${OLD_DAY_FEATURE_WEIGHT}" \
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

    "${PYTHON_BIN}" tools/stage86_lora_balanced_old_new_training_report.py \
        --save-dir "${SAVE_DIR}" \
        --job-id "${JOB_ID}" \
        --old-day-ce-weight "${OLD_DAY_CE_WEIGHT}" \
        --old-day-feature-weight "${OLD_DAY_FEATURE_WEIGHT}" \
        --margin-weight "${MARGIN_WEIGHT}" \
        --margin "${MARGIN_VALUE}" \
        --output "${ROOT}/STAGE86_LORA_BALANCED_OLD_NEW_TRAINING_SEED7_REPORT_${JOB_ID}.md" \
        --summary-json "${ROOT}/stage86_lora_balanced_old_new_training_summary_${JOB_ID}.json"
} >"${RUN_LOG}" 2>&1
