#!/bin/bash
# Stage 51：LoRa 新类吸收 seed31 小矩阵。
#
# 背景：
#   Stage48 已把 LoRa 当前正式候选提升到约 28%，但 seed31 的 New 仍偏低；
#   Stage50 证明降低 old:new batch ratio 不是突破口。本阶段不再调聚类阈值
#   或旧类采样比例，而是直接验证“新伪类注册后是否被分类头充分吸收”。
#
# 验证内容：
#   base                : Stage48 seed31 配置复跑。
#   new_proto_w0p10     : 当前轮新伪类原型归属损失。
#   head_boost2         : head warmup 阶段提高当前新伪类 CE 相对权重。
#   imprint_scale1p5    : 新类 classifier imprint 权重尺度放大。
#
# 通过门槛：
#   R3 New 比 Stage48 seed31 至少 +0.03，Overall 不低于 Stage48 seed31，
#   Forgetting 恶化不超过 +0.05；否则不扩三种子。
#SBATCH --job-name=stage51_lora_absorb_s31
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage51/slurm_lora_new_absorption_%j.out
#SBATCH --error=results/stage51/slurm_lora_new_absorption_%j.err

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
ROOT="results/stage51"
S28_NPZ="${LORA25_S28_NPZ:-results/stage46/lora25_diffdays_indoor_raw1024_s28_46103936.npz}"
mkdir -p "${ROOT}"

echo "Stage51 data_root=${DATA_ROOT}"
echo "Stage51 s28_npz=${S28_NPZ}"
echo "Stage51 python=${PYTHON_BIN}"
if [ ! -f "${S28_NPZ}" ]; then
    echo "Missing s28 NPZ: ${S28_NPZ}. Run Stage46 first or pass LORA25_S28_NPZ." >&2
    exit 2
fi

run_variant() {
    local variant="$1"
    shift
    local save_dir="${ROOT}/lora_s51_${variant}_${JOB_ID}"
    local checkpoint="${save_dir}/closedset_lora25_lora_chirp_10known.pth"
    mkdir -p "${save_dir}"
    echo "== Stage51 variant ${variant}: save_dir=${save_dir} =="
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
        "$@"
}

run_variant base
run_variant new_proto_w0p10 --radcil_new_prototype_weight 0.10
run_variant head_boost2 --radcil_new_head_boost 2.0
run_variant imprint_scale1p5 --radcil_new_imprint_scale 1.5

"${PYTHON_BIN}" tools/stage51_lora_new_class_absorption_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --variants base new_proto_w0p10 head_boost2 imprint_scale1p5 \
    --output "${ROOT}/STAGE51_LORA_NEW_CLASS_ABSORPTION_SEED31_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage51_lora_new_class_absorption_summary_${JOB_ID}.json"
