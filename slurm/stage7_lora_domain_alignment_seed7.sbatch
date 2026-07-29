#!/bin/bash
# 阶段 7：LoRa 训练期跨天特征分布对齐 seed7 二元验证。
#
# 只比较关闭/开启 CORAL-style discovery/replay 分布对齐，固定
# SupCon 表征、RADCIL 配比和 strict 协议；held-out 只做最终审计。

#SBATCH --job-name=osei-s7-lora-domain-s7
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage7-lora-domain-s7-%j.out
#SBATCH --error=results/slurm-osei-stage7-lora-domain-s7-%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
MATRIX_ROOT="results/stage7/lora_domain_alignment_seed7_${JOB_ID}"
CHECKPOINT="${LORA_DOMAIN_ALIGNMENT_CHECKPOINT:-results/stage5/lora_representation_screen_seed7_44573249/supcon_rf_aug.pth}"
LORA_NPZ="${LORA25_NPZ:-datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"

for VARIANT in alignment_off alignment_on; do
    EXTRA_ARGS=()
    if [[ "${VARIANT}" == "alignment_on" ]]; then
        EXTRA_ARGS+=(--radcil_domain_alignment_weight 0.05)
    fi
    "${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
        --dataset_path "${LORA_NPZ}" \
        --save_dir "${MATRIX_ROOT}/${VARIANT}" \
        --checkpoint "${CHECKPOINT}" \
        --seed 7 \
        --initial_known_classes 10 \
        --round_size 5 \
        --num_rounds 3 \
        --use_supcon \
        --supcon_weight 0.1 \
        --disable_cil_baselines \
        --disable_visualization \
        --cil_head_warmup_epochs 2 \
        --cil_joint_epochs 8 \
        --cil_replay_weight 3.0 \
        --radcil_old_new_batch_ratio 2.0 \
        --mvacc_enforce_round_size \
        --mvacc_min_cluster_ratio 0.03 \
        --mvacc_min_samples 5 \
        "${EXTRA_ARGS[@]}"
done

"${PYTHON_BIN}" tools/stage7_lora_domain_alignment_report.py \
    --matrix-root "${MATRIX_ROOT}" \
    --job-id "${JOB_ID}" \
    --output "results/stage7/STAGE7_LORA_DOMAIN_ALIGNMENT_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage7/stage7_lora_domain_alignment_seed7_summary_${JOB_ID}.json"
