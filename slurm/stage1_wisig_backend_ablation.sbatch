#!/bin/bash
# 阶段 1 WiSig CIL 后端消融任务。
#
# 用途：
#   1. 复用 Job 44420671 的 SupCon 表征方向，围绕 R3 旧类遗忘做短消融。
#   2. 逐项检查 KD、回放权重、回放容量、head-only 训练和低骨干学习率。
#   3. 每个变体只写入 results/stage1/wisig_backend_*，不覆盖历史结果。

#SBATCH --job-name=osei-stage1-backend
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage1-backend-%j.out
#SBATCH --error=results/slurm-osei-stage1-backend-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

mkdir -p results/stage1

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DATASET_PATH="${WISIG_DATASET_PATH:-数据集/WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl}"
JOB_ID="${SLURM_JOB_ID:-manual}"

"${PYTHON_BIN}" tools/stage1_backend_ablation_plan.py \
    --project-root "${PROJECT_ROOT}" \
    --output "results/stage1/stage1_backend_ablation_plan_${JOB_ID}.json"

"${PYTHON_BIN}" tools/stage1_discovery_adapter_contract.py \
    --project-root "${PROJECT_ROOT}" \
    --output "results/stage1/stage1_discovery_adapter_contract_${JOB_ID}.json"

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
    local save_dir="results/stage1/wisig_backend_${variant}_${JOB_ID}"
    mkdir -p "${save_dir}"
    echo "运行阶段 1 WiSig 后端消融变体：${variant}"
    "${PYTHON_BIN}" experiments/exp_wisig_mvacc_cil_strict.py \
        "${COMMON_ARGS[@]}" \
        --save_dir "${save_dir}" \
        "$@"
}

run_variant "supcon_default_backend"
run_variant "kd_off" --cil_kd_weight 0.0
run_variant "replay_x2" --cil_replay_weight 2.0
run_variant "memory_50" --memory_per_class 50
run_variant "head_only" --cil_joint_epochs 0
run_variant "low_backbone_lr" --cil_backbone_lr 1e-6

echo "阶段 1 WiSig 后端消融完成：${PROJECT_ROOT}/results/stage1"
