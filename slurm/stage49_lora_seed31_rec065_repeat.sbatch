#!/bin/bash
# Stage 49：LoRa seed31 rec065 复现稳定性验证。
#
# 背景：
#   Stage47 seed31 rec065 曾把 R3 New 修到 0.4690，但 Stage48 同为 seed31
#   的 rec065 只有 0.4262。两者参数接近，优先怀疑端到端 GPU 训练存在
#   同 seed 波动，而不是马上继续扩大算法搜索。
#
# 验证目标：
#   1) 使用已补齐确定性设置的主入口，固定 seed31 和 rec065 配置连续跑 3 次。
#   2) 判断 R3 Overall/New/Forgetting 是否稳定；若仍大幅波动，先归档为
#      复现风险，不把单次高 New 当作正式提升。
#   3) 若三次都稳定优于 Stage48 seed31，才考虑把确定性修复后结果扩三种子。
#SBATCH --job-name=stage49_lora_s31_repeat
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage49/slurm_lora_seed31_rec065_repeat_%j.out
#SBATCH --error=results/stage49/slurm_lora_seed31_rec065_repeat_%j.err

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
ROOT="results/stage49"
S28_NPZ="${LORA25_S28_NPZ:-results/stage46/lora25_diffdays_indoor_raw1024_s28_46103936.npz}"
mkdir -p "${ROOT}"

echo "Stage49 data_root=${DATA_ROOT}"
echo "Stage49 s28_npz=${S28_NPZ}"
echo "Stage49 python=${PYTHON_BIN}"
if [ ! -f "${S28_NPZ}" ]; then
    echo "Missing s28 NPZ: ${S28_NPZ}. Run Stage46 first or pass LORA25_S28_NPZ." >&2
    exit 2
fi

run_repeat() {
    local repeat_id="$1"
    local save_dir="${ROOT}/lora_seed31_rec065_repeat${repeat_id}_${JOB_ID}"
    local checkpoint="${save_dir}/closedset_lora25_lora_chirp_10known.pth"
    mkdir -p "${save_dir}"
    echo "== Stage49 repeat ${repeat_id}: save_dir=${save_dir} =="
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
        --radcil_old_new_batch_ratio 2.0
}

run_repeat 1
run_repeat 2
run_repeat 3

"${PYTHON_BIN}" tools/stage49_lora_seed31_rec065_repeat_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --repeats 1 2 3 \
    --output "${ROOT}/STAGE49_LORA_SEED31_REC065_REPEAT_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage49_lora_seed31_rec065_repeat_summary_${JOB_ID}.json"
