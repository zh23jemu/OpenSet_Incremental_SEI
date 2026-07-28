#!/bin/bash
# 阶段 5：LoRa seed7 目标簇数约束 + 分组双头融合短验证。
#
# 本作业复用 IQ_7 预选的 supcon_rf_aug checkpoint。融合权重仅从 IQ_7
# 候选集合选择；当前轮新类始终保留 RADCIL 网络头，旧类才融合 DOI replay
# 原型。MV-ACC 只按协议预先声明的每轮 5 类约束过聚类，不读取发现真值。
# 可在提交时使用 `sbatch --partition=其他CPU分区 本脚本` 覆盖默认分区。

#SBATCH --job-name=osei-stage5-lora-hybrid-s7
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=results/slurm-osei-stage5-lora-hybrid-s7-%j.out
#SBATCH --error=results/slurm-osei-stage5-lora-hybrid-s7-%j.err

set -euo pipefail
module load slurm
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
JOB_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="results/stage5/lora_grouped_hybrid_seed7_${JOB_ID}"
CHECKPOINT="results/stage5/lora_representation_screen_seed7_44573249/supcon_rf_aug.pth"
LORA_NPZ="${LORA25_NPZ:-datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"

"${PYTHON_BIN}" experiments/exp_lora25_mvacc_cil_strict.py \
    --dataset_path "${LORA_NPZ}" \
    --save_dir "${OUTPUT_DIR}" \
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
    --radcil_grouped_doi_fusion \
    --radcil_doi_fusion_candidates "0,0.25,0.5,0.75,1.0" \
    --radcil_doi_align_lambda 0.30 \
    --radcil_doi_prototype_temperature 0.10 \
    --mvacc_enforce_round_size \
    --mvacc_min_cluster_ratio 0.03 \
    --mvacc_min_samples 5

"${PYTHON_BIN}" tools/stage5_lora_grouped_hybrid_report.py \
    --save-dir "${OUTPUT_DIR}" \
    --job-id "${JOB_ID}" \
    --output "results/stage5/STAGE5_LORA_GROUPED_HYBRID_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage5/stage5_lora_grouped_hybrid_seed7_summary_${JOB_ID}.json"
