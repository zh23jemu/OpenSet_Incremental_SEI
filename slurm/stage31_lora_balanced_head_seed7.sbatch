#!/usr/bin/env bash
# Stage 31：LoRa Chirp + GPCC 下的类均衡分类头重校准 seed7 短矩阵。
#
# 该实验只比较重校准轮数 0/1/3。0 是同配置基线；1/3 在每轮 RADCIL
# 完成后冻结 backbone，使用当前轮伪标签和 replay memory 按类等量抽样，
# 短暂更新分类头。它不读取 held-out IQ_8-10 真值，也不改变 strict split。
# 结果报告只使用小型 CSV/JSON/日志，模型与 replay 二进制留在计算节点。
#
# 提交方式：
#   sbatch --account=gpo-ifv7xx --qos=shortjobs --partition=gpuHz \
#       --time=01:00:00 slurm/stage31_lora_balanced_head_seed7.sbatch

#SBATCH --job-name=osei-s31-lora
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=results/stage31/slurm_lora_balanced_head_seed7_%j.out
#SBATCH --error=results/stage31/slurm_lora_balanced_head_seed7_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
# 独立 Git worktree 通常只包含代码，不重复复制虚拟环境；优先复用主项目
# 已通过 CUDA 自检的环境，但所有 Python 模块仍从当前 worktree 加载。
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python"
fi
JOB_ID="${SLURM_JOB_ID}"
mkdir -p "${PROJECT_ROOT}/results/stage31"
cd "${PROJECT_ROOT}"

for recalibration_epochs in 0 1 3; do
    SAVE_DIR="results/stage31/lora_balanced_head_seed7_e${recalibration_epochs}_${JOB_ID}"
    mkdir -p "${SAVE_DIR}"
    "${PYTHON_BIN}" experiments/exp_wisig_mvacc_cil_strict.py \
        --dataset_profile lora25 \
        --dataset_path datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz \
        --seed 7 \
        --closedset_backbone lora_chirp \
        --discovery_backend gpcc \
        --cross_day_repr_adaptation \
        --lora_ssl_adaptation \
        --enable_joint_discovery_refinement \
        --radcil_balanced_head_recalibration_epochs "${recalibration_epochs}" \
        --save_dir "${SAVE_DIR}"
done

"${PYTHON_BIN}" tools/stage31_lora_balanced_head_report.py \
    --root results/stage31 \
    --job-id "${JOB_ID}" \
    --output "results/stage31/STAGE31_LORA_BALANCED_HEAD_SEED7_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage31/stage31_lora_balanced_head_seed7_summary_${JOB_ID}.json"
