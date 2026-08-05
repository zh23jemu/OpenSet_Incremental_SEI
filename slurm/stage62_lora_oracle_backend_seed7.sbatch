#!/bin/bash
# Stage 62：LoRa oracle discovery 下的后端保护诊断 seed7。
#
# 注意：
#   该实验读取当前 discovery 真值生成完美簇，只用于瓶颈定位。
#   结果不能作为正式方法、baseline 或论文主表指标。
#
# 诊断问题：
#   Stage61 证明完美 discovery 主要抬 New，但 Old/Forgetting 仍限制 Overall。
#   本阶段固定 oracle discovery，比较旧类原型路由和 grouped DOI-style fusion
#   能否在不压塌 New 的情况下救回 Old。
#SBATCH --job-name=stage62_lora_oracle_backend
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage62/slurm_lora_oracle_backend_seed7_%j.out
#SBATCH --error=results/stage62/slurm_lora_oracle_backend_seed7_%j.err

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
ROOT="results/stage62"
S28_NPZ="${LORA25_S28_NPZ:-results/stage46/lora25_diffdays_indoor_raw1024_s28_46103936.npz}"
mkdir -p "${ROOT}"

echo "Stage62 data_root=${DATA_ROOT}"
echo "Stage62 s28_npz=${S28_NPZ}"
echo "Stage62 python=${PYTHON_BIN}"
if [ ! -f "${S28_NPZ}" ]; then
    echo "Missing s28 NPZ: ${S28_NPZ}. Run Stage46 first or pass LORA25_S28_NPZ." >&2
    exit 2
fi

run_variant() {
    local variant="$1"
    shift
    local save_dir="${ROOT}/lora_s62_${variant}_${JOB_ID}"
    local checkpoint="${save_dir}/closedset_lora25_lora_chirp_10known.pth"
    mkdir -p "${save_dir}"
    echo "== Stage62 variant ${variant}: save_dir=${save_dir} =="
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
        --discovery_backend oracle \
        --oracle_discovery_diagnostic \
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

run_variant oracle_base
run_variant oracle_oldroute \
    --radcil_old_prototype_route \
    --radcil_old_prototype_route_candidates 0.55,0.65,0.75,0.85,0.95
run_variant oracle_grouped_doi \
    --radcil_grouped_doi_fusion \
    --radcil_doi_fusion_candidates 0.10,0.25,0.50,0.75

"${PYTHON_BIN}" tools/stage62_lora_oracle_backend_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --variants oracle_base oracle_oldroute oracle_grouped_doi \
    --output "${ROOT}/STAGE62_LORA_ORACLE_BACKEND_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage62_lora_oracle_backend_summary_${JOB_ID}.json"
