#!/bin/bash
# Stage 61：LoRa oracle discovery 上界诊断 seed7。
#
# 注意：
#   该实验显式读取当前 discovery 真值生成完美簇，只用于瓶颈定位。
#   结果不能作为正式方法、baseline 或论文主表指标。
#
# 诊断问题：
#   如果 discovery 聚类完美，Stage48 的 LoRaChirp + joint discovery-CIL/RADCIL
#   后端能否明显超过当前 28%？若仍然低，说明瓶颈不只是聚类，而是
#   跨天表征/增量后端本身。
#SBATCH --job-name=stage61_lora_oracle
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage61/slurm_lora_oracle_discovery_seed7_%j.out
#SBATCH --error=results/stage61/slurm_lora_oracle_discovery_seed7_%j.err

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
ROOT="results/stage61"
S28_NPZ="${LORA25_S28_NPZ:-results/stage46/lora25_diffdays_indoor_raw1024_s28_46103936.npz}"
SAVE_DIR="${ROOT}/lora_s61_oracle_seed7_${JOB_ID}"
CHECKPOINT="${SAVE_DIR}/closedset_lora25_lora_chirp_10known.pth"
mkdir -p "${SAVE_DIR}"

echo "Stage61 data_root=${DATA_ROOT}"
echo "Stage61 s28_npz=${S28_NPZ}"
echo "Stage61 python=${PYTHON_BIN}"
echo "Stage61 save_dir=${SAVE_DIR}"
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
    --radcil_old_new_batch_ratio 2.0

"${PYTHON_BIN}" tools/stage61_lora_oracle_discovery_report.py \
    --save-dir "${SAVE_DIR}" \
    --job-id "${JOB_ID}"
