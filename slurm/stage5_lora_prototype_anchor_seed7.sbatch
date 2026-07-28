#!/bin/bash
# 阶段 5：LoRa 训练期旧类原型锚定 seed7 预注册矩阵。
#
# 三个候选仅改变类中心锚定权重。报告器先按 IQ_7 的 R3 旧类保持率选择，
# 然后才展示 IQ_8-10 held-out 事后指标。所有变体复用同一个预选 checkpoint，
# 且使用协议目标 5 簇约束，避免再次混入过聚类差异。

#SBATCH --job-name=osei-stage5-lora-anchor-s7
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=results/slurm-osei-stage5-lora-anchor-s7-%j.out
#SBATCH --error=results/slurm-osei-stage5-lora-anchor-s7-%j.err

set -euo pipefail
module load slurm
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
JOB_ID="${SLURM_JOB_ID:-manual}"
MATRIX_ROOT="results/stage5/lora_prototype_anchor_seed7_${JOB_ID}"
CHECKPOINT="results/stage5/lora_representation_screen_seed7_44573249/supcon_rf_aug.pth"
LORA_NPZ="${LORA25_NPZ:-datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"

for WEIGHT in 0.0 0.25 1.0; do
    VARIANT="anchor_${WEIGHT//./p}"
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
        --radcil_prototype_anchor_weight "${WEIGHT}" \
        --mvacc_enforce_round_size \
        --mvacc_min_cluster_ratio 0.03 \
        --mvacc_min_samples 5
done

"${PYTHON_BIN}" tools/stage5_lora_prototype_anchor_report.py \
    --matrix-root "${MATRIX_ROOT}" \
    --job-id "${JOB_ID}" \
    --output "results/stage5/STAGE5_LORA_PROTOTYPE_ANCHOR_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage5/stage5_lora_prototype_anchor_seed7_summary_${JOB_ID}.json"

# 只读生成服务器产物清单；不删除、移动或修改任何已有运行产物。
"${PYTHON_BIN}" tools/slurm_artifact_inventory.py \
    --results-root results \
    --output "results/stage5/slurm_artifact_inventory_${JOB_ID}.json"
