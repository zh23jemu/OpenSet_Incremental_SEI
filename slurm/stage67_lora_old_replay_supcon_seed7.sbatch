#!/bin/bash
# Stage 67：LoRa 旧类 replay SupCon seed7 小矩阵。
#
# Stage66 已证明 LoRa R3 Overall 主要被旧类跨天保持拖住：旧类占 80%
# 权重，当前 Old 约 0.23。这里不再继续抬 New，也不改变 discovery 前端，
# 只在 RADCIL 训练期对 replay 旧类样本加入监督对比保持项。
#
# 实验边界：
#   - 固定 Stage48 raw s28 + recording-consensus 0.65 主配置。
#   - 共享同一个 seed7 closed-set checkpoint，避免把初始训练波动当成收益。
#   - 权重 0 是同 job 对照；0.05/0.10 是最小风险验证。
#   - 只使用 replay 训练标签，不读取未知真值或 held-out eval 选参。
#SBATCH --job-name=stage67_lora_oldsupcon
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage67/slurm_lora_old_replay_supcon_seed7_%j.out
#SBATCH --error=results/stage67/slurm_lora_old_replay_supcon_seed7_%j.err

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
ROOT="results/stage67"
S28_NPZ="${LORA25_S28_NPZ:-results/stage46/lora25_diffdays_indoor_raw1024_s28_46103936.npz}"
CHECKPOINT="${ROOT}/closedset_lora25_lora_chirp_10known_seed7_${JOB_ID}.pth"
mkdir -p "${ROOT}"

echo "Stage67 s28_npz=${S28_NPZ}"
echo "Stage67 checkpoint=${CHECKPOINT}"
echo "Stage67 python=${PYTHON_BIN}"
if [ ! -f "${S28_NPZ}" ]; then
    echo "Missing s28 NPZ: ${S28_NPZ}. Run Stage46 first or pass LORA25_S28_NPZ." >&2
    exit 2
fi

run_variant() {
    local label="$1"
    local weight="$2"
    local train_flag="$3"
    local save_dir="${ROOT}/lora_s67_oldreplay_supcon_${label}_seed7_${JOB_ID}"
    mkdir -p "${save_dir}"

    local maybe_train=()
    if [ "${train_flag}" = "train" ]; then
        maybe_train=(--train_closedset --epochs 20)
    fi

    "${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
        --dataset_profile lora25 \
        --dataset_path "${S28_NPZ}" \
        --save_dir "${save_dir}" \
        --checkpoint "${CHECKPOINT}" \
        "${maybe_train[@]}" \
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
        --radcil_old_replay_supcon_weight "${weight}"
}

run_variant "w0p00" "0.00" "train"
run_variant "w0p05" "0.05" "load"
run_variant "w0p10" "0.10" "load"

"${PYTHON_BIN}" tools/stage67_lora_old_replay_supcon_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --output "${ROOT}/STAGE67_LORA_OLD_REPLAY_SUPCON_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage67_lora_old_replay_supcon_summary_${JOB_ID}.json"
