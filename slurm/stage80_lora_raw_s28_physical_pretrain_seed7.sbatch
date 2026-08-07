#!/bin/bash
# Stage80：LoRa Setup 1 原始 I/Q s28 + 物理域预训练 seed7。
#
# 目的：
#   Stage46 说明 raw-s28 能提升 Overall/Old，但 New 稳定性不足；Stage38
#   说明物理域预训练有过 seed7 正信号但三种子不稳。本脚本把两者合并：
#   先从 Setup1 原始 I/Q 生成 raw1024_s28 strict NPZ，再用该 NPZ 训练
#   物理域 checkpoint，最后在同一 strict split 下跑 GPCC + RADCIL。
#
# 协议边界：
#   - 原始 I/Q 只用于生成协议内 IQ_1-7 discovery/train 和 IQ_8-10 eval。
#   - 预训练只使用 Day1 train/validation；不读取 held-out eval 真值。
#   - 后续 CIL 与 Stage46 seed7 配置保持一致，方便公平比较。

#SBATCH --job-name=stage80_lora_raw_phys
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage80/slurm_lora_raw_s28_physical_seed7_%j.out
#SBATCH --error=results/stage80/slurm_lora_raw_s28_physical_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
ROOT="results/stage80"
RAW_DIR="${RAW_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage44/raw_setup1_iq}"
CACHE_DIR="${LORA25_CACHE_DIR:-${ROOT}/cache_raw_setup1_s28_${JOB_ID}}"
RAW_PROBE_NPZ="${ROOT}/lora25_raw_probe_s28_${JOB_ID}.npz"
SOURCE_MANIFEST="${ROOT}/lora25_source_manifest_raw_s28_${JOB_ID}.json"
S28_NPZ="${LORA25_S28_NPZ:-${ROOT}/lora25_diffdays_indoor_raw1024_s28_${JOB_ID}.npz}"
ALIGN_MANIFEST="${ROOT}/lora25_alignment_manifest_raw1024_s28_${JOB_ID}.json"
CHECKPOINT="${ROOT}/lora_raw_s28_physical_checkpoint_seed7_${JOB_ID}.pth"
SAVE_DIR="${ROOT}/lora_raw_s28_physical_pretrain_seed7_${JOB_ID}"

cd "${PROJECT_ROOT}"
mkdir -p "${ROOT}" "${CACHE_DIR}" "${SAVE_DIR}"

echo "Stage80 project_root=${PROJECT_ROOT}"
echo "Stage80 python=${PYTHON_BIN}"
echo "Stage80 raw_dir=${RAW_DIR}"
echo "Stage80 s28_npz=${S28_NPZ}"

# 生成 raw-s28 所需的本地 range cache。若调用方已通过 LORA25_S28_NPZ
# 指向现有 strict NPZ，则跳过重建，减少排队后实际运行时间。
if [ ! -f "${S28_NPZ}" ]; then
    if [ ! -f "${RAW_PROBE_NPZ}" ]; then
        "${PYTHON_BIN}" datasets/lora25_compact/tools/build_lora25_compact.py \
            --output "${RAW_PROBE_NPZ}" \
            --cache_dir "${CACHE_DIR}" \
            --manifest "${SOURCE_MANIFEST}" \
            --raw_dir "${RAW_DIR}" \
            --windows_per_transmission 28
    fi

    "${PYTHON_BIN}" datasets/lora25_compact/tools/build_lora25_aligned.py \
        --cache_dir "${CACHE_DIR}" \
        --output "${S28_NPZ}" \
        --alignment_manifest "${ALIGN_MANIFEST}" \
        --symbols_per_transmission 28 \
        --target_len 1024 \
        --decimation 1 \
        --representation raw
fi

# 在 raw-s28 数据上训练 LoRa 物理域预训练 checkpoint。这里不启用 discovery
# unlabeled 或 domain adversarial，避免重复 Stage39/40 中已失败的方向。
"${PYTHON_BIN}" tools/stage38_lora_physical_pretrain.py \
    --dataset-path "${S28_NPZ}" \
    --output "${CHECKPOINT}" \
    --summary-json "${ROOT}/stage80_lora_raw_s28_physical_checkpoint_summary_${JOB_ID}.json" \
    --seed 7 \
    --epochs 20 \
    --batch-size 128 \
    --lr 1e-3 \
    --supcon-weight 0.1 \
    --supcon-temperature 0.2 \
    --physical-weight 0.10

# CIL 配置对齐 Stage46 raw-s28 seed7：GPCC、cross-day、LoRa SSL、joint
# refinement、replay 和 old:new batch ratio 均保持一致，只替换初始 checkpoint。
"${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
    --dataset_profile lora25 \
    --dataset_path "${S28_NPZ}" \
    --save_dir "${SAVE_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --closedset_backbone lora_chirp \
    --seed 7 --initial_known_classes 10 --round_size 5 --num_rounds 3 \
    --use_supcon --supcon_weight 0.1 \
    --disable_cil_baselines --disable_visualization \
    --discovery_backend gpcc \
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
    --cil_replay_weight 3.0 \
    --radcil_old_new_batch_ratio 2.0 \
    --memory_per_class 20

"${PYTHON_BIN}" tools/stage80_lora_raw_s28_physical_pretrain_report.py \
    --root "${ROOT}" \
    --job-id "${JOB_ID}" \
    --output "${ROOT}/STAGE80_LORA_RAW_S28_PHYSICAL_PRETRAIN_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "${ROOT}/stage80_lora_raw_s28_physical_pretrain_seed7_summary_${JOB_ID}.json"

echo "Stage80 LoRa raw-s28 + 物理域预训练 seed7 完成。"
