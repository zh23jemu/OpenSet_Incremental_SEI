#!/bin/bash
# 阶段 1 WiSig RADCIL 后端二阶矩阵任务。
#
# 用途：
#   1. 复用 Job 44420869 的 replay_x2 结论作为锚点；
#   2. 验证旧/新 batch 配比、masked KD、KD schedule、特征蒸馏和解冻范围；
#   3. 每个变体写入独立 results/stage1/wisig_radcil_* 目录，避免覆盖历史结果。
#
# 资源说明：
#   默认使用 gpuHz + shortjobs，目标是在 1 小时内完成单种子短验证。
#   如预计排队或运行时间更长，可提交时覆盖：
#   sbatch --partition=gpu --qos=normal --time=03:00:00 slurm/stage1_wisig_radcil_backend_matrix.sbatch

#SBATCH --job-name=osei-stage1-radcil
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage1-radcil-%j.out
#SBATCH --error=results/slurm-osei-stage1-radcil-%j.err

set -euo pipefail

module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

mkdir -p results/stage1

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DATASET_PATH="${WISIG_DATASET_PATH:-数据集/WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl}"
JOB_ID="${SLURM_JOB_ID:-manual}"

"${PYTHON_BIN}" tools/stage1_radcil_backend_matrix.py \
    --project-root "${PROJECT_ROOT}" \
    --output "results/stage1/stage1_radcil_backend_matrix_${JOB_ID}.json"

COMMON_ARGS=(
    --dataset_path "${DATASET_PATH}"
    --selected_rx_list 2
    --initial_known_classes 10
    --round_size 10
    --num_rounds 3
    --seed 7
    --train_closedset
    --epochs 8
    --batch_size 128
    --test_batch_size 256
    --disable_visualization
    --disable_cil_baselines
    --use_supcon
    --supcon_weight 0.1
    --cil_head_warmup_epochs 1
    --cil_joint_epochs 3
)

run_variant() {
    local variant="$1"
    shift
    local save_dir="results/stage1/wisig_radcil_${variant}_${JOB_ID}"
    mkdir -p "${save_dir}"
    echo "运行阶段 1 WiSig RADCIL 二阶变体：${variant}"
    "${PYTHON_BIN}" experiments/exp_wisig_mvacc_cil_strict.py \
        "${COMMON_ARGS[@]}" \
        --save_dir "${save_dir}" \
        "$@"
}

run_variant "replay_x2_confirm" --cil_replay_weight 2.0
run_variant "replay_x3" --cil_replay_weight 3.0
run_variant "balanced_old_new_batch" --cil_replay_weight 2.0 --radcil_old_new_batch_ratio 2.0
run_variant "masked_kd_low" --cil_replay_weight 2.0 --cil_kd_weight 0.25 --radcil_masked_kd
run_variant "masked_kd_schedule" --cil_replay_weight 2.0 --cil_kd_weight 0.5 --radcil_masked_kd --radcil_kd_schedule cosine --cil_temperature 3.0
run_variant "feature_distill_tail" --cil_replay_weight 2.0 --radcil_feature_distill_weight 0.5
run_variant "tail_unfreeze_plus_feature_distill" --cil_replay_weight 2.0 --cil_backbone_lr 5e-6 --radcil_feature_distill_weight 0.5 --radcil_unfreeze_scope tail

echo "阶段 1 WiSig RADCIL 后端二阶矩阵完成：${PROJECT_ROOT}/results/stage1"
