#!/bin/bash
# 阶段 1 WiSig 单种子短实验。
#
# 用途：
#   1. 在同一 WiSig strict 10+10x3 协议下比较轻量 CE 初始骨干和 SupCon 初始表征。
#   2. 使用较短 closed-set 训练和较短 CIL 微调，快速判断旧类遗忘是否主要来自表征质量。
#   3. 只写入 results/stage1/ 下的新实验结果，不删除、不覆盖历史结果。
#
# 提交示例：
#   sbatch slurm/stage1_wisig_short_screen.sbatch
#   sbatch --partition=gpuHz slurm/stage1_wisig_short_screen.sbatch

#SBATCH --job-name=osei-stage1-wisig
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage1-wisig-%j.out
#SBATCH --error=results/slurm-osei-stage1-wisig-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

mkdir -p results/stage1

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DATASET_PATH="${WISIG_DATASET_PATH:-数据集/WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl}"
JOB_ID="${SLURM_JOB_ID:-manual}"

# 先生成方法筛选报告，确保本次短实验的输入假设和候选矩阵可追溯。
"${PYTHON_BIN}" tools/stage1_method_screen.py \
    --project-root "${PROJECT_ROOT}" \
    --output "results/stage1/stage1_method_screen_${JOB_ID}.json"

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
    --cil_head_warmup_epochs 1
    --cil_joint_epochs 3
)

run_variant() {
    local variant="$1"
    shift
    local save_dir="results/stage1/wisig_short_${variant}_${JOB_ID}"
    mkdir -p "${save_dir}"
    echo "运行阶段 1 WiSig 短实验变体：${variant}"
    "${PYTHON_BIN}" experiments/exp_wisig_mvacc_cil_strict.py \
        "${COMMON_ARGS[@]}" \
        --save_dir "${save_dir}" \
        "$@"
}

run_variant "ce_baseline"
run_variant "supcon_representation" --use_supcon --supcon_weight 0.1

echo "阶段 1 WiSig 短实验完成：${PROJECT_ROOT}/results/stage1"
