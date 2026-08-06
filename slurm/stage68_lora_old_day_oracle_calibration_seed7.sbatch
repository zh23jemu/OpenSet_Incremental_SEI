#!/bin/bash
# Stage 68：LoRa 旧类跨天同日校准 oracle 诊断。
#
# 背景：
#   Stage66 已证明 R3 Overall 主要被旧类 Old 主导；Stage67 又证明 replay-only
#   old SupCon 几乎不抬 Old。本阶段不再加训练 loss，而是复跑当前 LoRa 正式
#   候选 Stage48 seed7，并保存 held-out eval dumps，离线计算“如果 Day2-4 旧类
#   有少量同日标注校准样本，Old/Overall 理论上能不能上去”。
#
# 重要边界：
#   1) oracle 报告读取 held-out 真值，只用于瓶颈诊断；
#   2) 不把该结果包装为正式方法提升；
#   3) 不扩三种子，除非诊断证明旧类同日校准是明确突破口。
#SBATCH --job-name=stage68_lora_old_oracle
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/stage68/slurm_lora_old_day_oracle_seed7_%j.out
#SBATCH --error=results/stage68/slurm_lora_old_day_oracle_seed7_%j.err

set -euo pipefail

module load slurm

DATA_ROOT="${DATA_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    # 远端轻量 worktree 通常不复制 .venv，回退到主项目已验证环境。
    PYTHON_BIN="/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python"
fi
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 127
fi

JOB_ID="${SLURM_JOB_ID:-manual}"
ROOT="results/stage68"
S28_NPZ="${LORA25_S28_NPZ:-results/stage46/lora25_diffdays_indoor_raw1024_s28_46103936.npz}"
SAVE_DIR="${ROOT}/lora_s68_stage48_dump_seed7_${JOB_ID}"
CHECKPOINT="${SAVE_DIR}/closedset_lora25_lora_chirp_10known.pth"

mkdir -p "${ROOT}" "${SAVE_DIR}"

echo "Stage68 data_root=${DATA_ROOT}"
echo "Stage68 s28_npz=${S28_NPZ}"
echo "Stage68 python=${PYTHON_BIN}"
echo "Stage68 save_dir=${SAVE_DIR}"
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
    --save_end_to_end_eval_dumps \
    --cil_head_warmup_epochs 2 \
    --cil_joint_epochs 8 \
    --incremental_batch_size 96 \
    --test_batch_size 192 \
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0

"${PYTHON_BIN}" tools/stage68_lora_old_day_oracle_calibration_report.py \
    --save-dir "${SAVE_DIR}" \
    --job-id "${JOB_ID}" \
    --output "${ROOT}/STAGE68_LORA_OLD_DAY_ORACLE_CALIBRATION_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage68_lora_old_day_oracle_calibration_summary_${JOB_ID}.json"
