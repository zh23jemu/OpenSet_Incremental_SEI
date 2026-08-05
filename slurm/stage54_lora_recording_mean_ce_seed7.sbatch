#!/bin/bash
# Stage 54：LoRa recording-level mean-logit CE seed7 矩阵。
#
# 背景：
#   Stage53 的 recording logits 一致性没有过门槛，说明“让同一 recording 的
#   symbol 预测彼此接近”还不够。本阶段进一步把同一 recording、同一伪类
#   的 symbol logits 聚合后直接做 CE，验证更大粒度训练监督是否能提升
#   symbol-level CIL。
#
# 验证内容：
#   base    : Stage48 seed7 配置复跑。
#   rce_w0p25: 当前轮 discovery 的 recording mean-logit CE，权重 0.25。
#   rce_w0p50: 当前轮 discovery 的 recording mean-logit CE，权重 0.50。
#
# 通过门槛：
#   相对 Stage48 seed7，R3 Overall 至少 +0.01；Old 不低于 Stage48 seed7；
#   New 下降不超过 0.02；Forgetting 恶化不超过 0.02。否则不扩三种子。
#SBATCH --job-name=stage54_lora_rec_ce_s7
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage54/slurm_lora_recording_mean_ce_seed7_%j.out
#SBATCH --error=results/stage54/slurm_lora_recording_mean_ce_seed7_%j.err

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
ROOT="results/stage54"
S28_NPZ="${LORA25_S28_NPZ:-results/stage46/lora25_diffdays_indoor_raw1024_s28_46103936.npz}"
mkdir -p "${ROOT}"

echo "Stage54 data_root=${DATA_ROOT}"
echo "Stage54 s28_npz=${S28_NPZ}"
echo "Stage54 python=${PYTHON_BIN}"
if [ ! -f "${S28_NPZ}" ]; then
    echo "Missing s28 NPZ: ${S28_NPZ}. Run Stage46 first or pass LORA25_S28_NPZ." >&2
    exit 2
fi

run_variant() {
    local variant="$1"
    shift
    local save_dir="${ROOT}/lora_s54_${variant}_${JOB_ID}"
    local checkpoint="${save_dir}/closedset_lora25_lora_chirp_10known.pth"
    mkdir -p "${save_dir}"
    echo "== Stage54 variant ${variant}: save_dir=${save_dir} =="
    "${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
        --dataset_profile lora25 \
        --dataset_path "${S28_NPZ}" \
        --save_dir "${save_dir}" \
        --checkpoint "${checkpoint}" \
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
        "$@"
}

run_variant base
run_variant rce_w0p25 --radcil_recording_ce_weight 0.25
run_variant rce_w0p50 --radcil_recording_ce_weight 0.50

"${PYTHON_BIN}" tools/stage54_lora_recording_mean_ce_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --variants base rce_w0p25 rce_w0p50 \
    --output "${ROOT}/STAGE54_LORA_RECORDING_MEAN_CE_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage54_lora_recording_mean_ce_summary_${JOB_ID}.json"
