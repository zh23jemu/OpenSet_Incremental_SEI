#!/usr/bin/env bash

# 阶段 0 的 GPU、依赖和数据完整性自检任务。
# 默认使用 gpuHz + shortjobs，控制在 1 小时以内；如需其它分区，可在提交时使用：
# sbatch --partition=gpu slurm/stage0_env_data_check.sbatch

#SBATCH --job-name=osei-stage0-check
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-%x-%j.out
#SBATCH --error=results/slurm-%x-%j.err

set -euo pipefail

# 以提交目录为项目根目录，避免把服务器绝对路径写死在脚本中。
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
OUTPUT_DIR="${PROJECT_ROOT}/results/stage0"
REPORT_PATH="${OUTPUT_DIR}/stage0_env_data_check_${SLURM_JOB_ID}.json"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "错误：未找到项目虚拟环境解释器 ${PYTHON_BIN}" >&2
    exit 2
fi

echo "项目目录：${PROJECT_ROOT}"
echo "计算节点：$(hostname)"
echo "任务编号：${SLURM_JOB_ID}"
nvidia-smi

# --hash-large-files 会校验从 GitHub Release 同步的四份原始数据，确保远端副本完整。
"${PYTHON_BIN}" tools/stage0_env_data_check.py \
    --project-root "${PROJECT_ROOT}" \
    --data-root "数据集" \
    --hash-large-files \
    --json | tee "${REPORT_PATH}"

echo "阶段 0 自检报告：${REPORT_PATH}"
