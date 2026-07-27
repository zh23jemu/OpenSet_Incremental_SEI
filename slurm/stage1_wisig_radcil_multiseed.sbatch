#!/bin/bash
# 阶段 1 WiSig RADCIL 多种子确认矩阵。
#
# 用途：
#   1. 只确认 ratio/weight 细化矩阵收敛出的两个候选；
#   2. 使用 seed 7/13/31 做 3 种子短实验；
#   3. 每个候选和种子写入独立 results/stage1/wisig_radcil_multiseed_* 目录。
#
# 分区可在提交时覆盖，例如：
#   sbatch --partition=gpu slurm/stage1_wisig_radcil_multiseed.sbatch

#SBATCH --job-name=osei-stage1-mseed
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage1-mseed-%j.out
#SBATCH --error=results/slurm-osei-stage1-mseed-%j.err

set -euo pipefail

module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

mkdir -p results/stage1

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DATASET_PATH="${WISIG_DATASET_PATH:-数据集/WiSig_CrossDay_40Tx_3Rx_4Day_300Sig_equalized.pkl}"
JOB_ID="${SLURM_JOB_ID:-manual}"

"${PYTHON_BIN}" tools/stage1_radcil_multiseed_plan.py \
    --project-root "${PROJECT_ROOT}" \
    --output "results/stage1/stage1_radcil_multiseed_plan_${JOB_ID}.json"

COMMON_ARGS=(
    --dataset_path "${DATASET_PATH}"
    --selected_rx_list 2
    --initial_known_classes 10
    --round_size 10
    --num_rounds 3
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

run_candidate() {
    local variant="$1"
    local ratio="$2"
    local replay_weight="$3"
    local seed="$4"
    local save_dir="results/stage1/wisig_radcil_multiseed_${variant}_seed${seed}_${JOB_ID}"

    mkdir -p "${save_dir}"
    echo "运行 RADCIL 多种子候选：variant=${variant}, seed=${seed}, ratio=${ratio}, replay_weight=${replay_weight}"
    "${PYTHON_BIN}" experiments/exp_wisig_mvacc_cil_strict.py \
        "${COMMON_ARGS[@]}" \
        --seed "${seed}" \
        --save_dir "${save_dir}" \
        --cil_replay_weight "${replay_weight}" \
        --radcil_old_new_batch_ratio "${ratio}"
}

for seed in 7 13 31; do
    run_candidate "ratio_3p0_replay_3p0" "3.0" "3.0" "${seed}"
    run_candidate "ratio_2p0_replay_3p0" "2.0" "3.0" "${seed}"
done

echo "阶段 1 WiSig RADCIL 多种子确认完成：${PROJECT_ROOT}/results/stage1"
