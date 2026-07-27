#!/bin/bash
# 阶段 1 WiSig RADCIL old:new ratio / replay weight 细化矩阵。
#
# 用途：
#   1. 以 Job 44422380 的 balanced_old_new_batch 为锚点；
#   2. 比较 old:new ratio 1.5/2.0/3.0 与 replay weight 2.0/2.5/3.0；
#   3. 每个变体写入独立 results/stage1/wisig_radcil_ratio_* 目录。

#SBATCH --job-name=osei-stage1-ratio
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage1-ratio-%j.out
#SBATCH --error=results/slurm-osei-stage1-ratio-%j.err

set -euo pipefail

module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

mkdir -p results/stage1

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DATASET_PATH="${WISIG_DATASET_PATH:-数据集/WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl}"
JOB_ID="${SLURM_JOB_ID:-manual}"

"${PYTHON_BIN}" tools/stage1_radcil_ratio_weight_matrix.py \
    --project-root "${PROJECT_ROOT}" \
    --output "results/stage1/stage1_radcil_ratio_weight_matrix_${JOB_ID}.json"

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
    local ratio="$1"
    local replay_weight="$2"
    local ratio_name="${ratio/./p}"
    local replay_name="${replay_weight/./p}"
    local variant="ratio_${ratio_name}_replay_${replay_name}"
    local save_dir="results/stage1/wisig_radcil_ratio_${variant}_${JOB_ID}"
    mkdir -p "${save_dir}"
    echo "运行 RADCIL ratio/weight 变体：${variant}"
    "${PYTHON_BIN}" experiments/exp_wisig_mvacc_cil_strict.py \
        "${COMMON_ARGS[@]}" \
        --save_dir "${save_dir}" \
        --cil_replay_weight "${replay_weight}" \
        --radcil_old_new_batch_ratio "${ratio}"
}

for ratio in 1.5 2.0 3.0; do
    for replay_weight in 2.0 2.5 3.0; do
        run_variant "${ratio}" "${replay_weight}"
    done
done

echo "阶段 1 WiSig RADCIL ratio/weight 细化矩阵完成：${PROJECT_ROOT}/results/stage1"
