#!/bin/bash
# Stage 57：LoRa top80 refined filter + old-prototype route seed7 验证。
#
# 背景：
#   Stage56 的 top80 能把 New 提高到 0.5667，但 Old/Forgetting 明显变差。
#   本阶段不再改训练损失，而是把 top80 作为 new-specialist，再用已有的
#   old-prototype-route 作为 old-stable 保护。路由阈值只用 Day1/IQ_7 旧类
#   验证集校准，held-out IQ_8-10 只做最终评估。
#
# 验证内容：
#   base        : Stage48 seed7 配置复跑。
#   filter_top080: Stage56 的新类吸收候选。
#   top080_oldroute: top80 + IQ_7 校准 old-prototype-route。
#
# 通过门槛：
#   相对 Stage48 seed7，R3 Overall 至少 +0.01；Old 不低于 Stage48 seed7；
#   New 下降不超过 0.02；Forgetting 恶化不超过 0.02。否则不扩三种子。
#SBATCH --job-name=stage57_lora_top80_route
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage57/slurm_lora_filter_oldroute_seed7_%j.out
#SBATCH --error=results/stage57/slurm_lora_filter_oldroute_seed7_%j.err

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
ROOT="results/stage57"
S28_NPZ="${LORA25_S28_NPZ:-results/stage46/lora25_diffdays_indoor_raw1024_s28_46103936.npz}"
mkdir -p "${ROOT}"

echo "Stage57 data_root=${DATA_ROOT}"
echo "Stage57 s28_npz=${S28_NPZ}"
echo "Stage57 python=${PYTHON_BIN}"
if [ ! -f "${S28_NPZ}" ]; then
    echo "Missing s28 NPZ: ${S28_NPZ}. Run Stage46 first or pass LORA25_S28_NPZ." >&2
    exit 2
fi

run_variant() {
    local variant="$1"
    shift
    local save_dir="${ROOT}/lora_s57_${variant}_${JOB_ID}"
    local checkpoint="${save_dir}/closedset_lora25_lora_chirp_10known.pth"
    mkdir -p "${save_dir}"
    echo "== Stage57 variant ${variant}: save_dir=${save_dir} =="
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
run_variant filter_top080 --radcil_refined_filter_top_fraction 0.80 --radcil_refined_filter_min_per_class 1
run_variant top080_oldroute \
    --radcil_refined_filter_top_fraction 0.80 \
    --radcil_refined_filter_min_per_class 1 \
    --radcil_old_prototype_route \
    --radcil_old_prototype_route_candidates 0.55,0.65,0.75,0.85,0.95

"${PYTHON_BIN}" tools/stage57_lora_filter_oldroute_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --variants base filter_top080 top080_oldroute \
    --output "${ROOT}/STAGE57_LORA_FILTER_OLDROUTE_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage57_lora_filter_oldroute_summary_${JOB_ID}.json"
