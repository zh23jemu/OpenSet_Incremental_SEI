#!/bin/bash
# Stage 53：LoRa 训练期 recording/transmission 级一致性 seed7 矩阵。
#
# 背景：
#   Stage48 已把 LoRa 正式候选提升到约 28%，但 symbol-level 绝对准确率仍低；
#   Stage51/52 说明继续调新类 head 或简单替换表示都不是突破口。本阶段利用
#   LoRa 数据天然的 recording_id：同一 recording 内的多个 symbol 片段应当
#   给出更一致的预测分布。
#
# 验证内容：
#   base     : Stage48 seed7 配置复跑。
#   rec_w0p10: 当前轮 discovery 中同 recording、同伪类 symbol 的 logits 一致性。
#   rec_w0p30: 更强的一致性约束。
#
# 通过门槛：
#   相对 Stage48 seed7，R3 Overall 至少 +0.01；Old 不低于 Stage48 seed7；
#   New 下降不超过 0.02；Forgetting 恶化不超过 0.02。否则不扩三种子。
#SBATCH --job-name=stage53_lora_rec_cons_s7
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage53/slurm_lora_recording_consistency_seed7_%j.out
#SBATCH --error=results/stage53/slurm_lora_recording_consistency_seed7_%j.err

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
ROOT="results/stage53"
S28_NPZ="${LORA25_S28_NPZ:-results/stage46/lora25_diffdays_indoor_raw1024_s28_46103936.npz}"
mkdir -p "${ROOT}"

echo "Stage53 data_root=${DATA_ROOT}"
echo "Stage53 s28_npz=${S28_NPZ}"
echo "Stage53 python=${PYTHON_BIN}"
if [ ! -f "${S28_NPZ}" ]; then
    echo "Missing s28 NPZ: ${S28_NPZ}. Run Stage46 first or pass LORA25_S28_NPZ." >&2
    exit 2
fi

run_variant() {
    local variant="$1"
    shift
    local save_dir="${ROOT}/lora_s53_${variant}_${JOB_ID}"
    local checkpoint="${save_dir}/closedset_lora25_lora_chirp_10known.pth"
    mkdir -p "${save_dir}"
    echo "== Stage53 variant ${variant}: save_dir=${save_dir} =="
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
run_variant rec_w0p10 --radcil_recording_consistency_weight 0.10
run_variant rec_w0p30 --radcil_recording_consistency_weight 0.30

"${PYTHON_BIN}" tools/stage53_lora_recording_consistency_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --variants base rec_w0p10 rec_w0p30 \
    --output "${ROOT}/STAGE53_LORA_RECORDING_CONSISTENCY_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage53_lora_recording_consistency_summary_${JOB_ID}.json"
