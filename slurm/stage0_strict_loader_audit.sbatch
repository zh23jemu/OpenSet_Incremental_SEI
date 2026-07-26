#!/bin/bash
# 阶段 0 strict loader 审计任务。
#
# 用途：
#   1. 在 Slurm 服务器上运行 WiSig / ADS-B strict loader。
#   2. 输出输入形状、类别数、样本数、split 边界和协议审计 JSON。
#   3. 只做数据完整性验证，不训练模型，不调参，不删除或移动数据。
#
# 分区说明：
#   这是 CPU 数据加载任务，默认使用 defq。若集群排队需要调整，可在提交时覆盖：
#   sbatch --partition=目标分区 slurm/stage0_strict_loader_audit.sbatch

#SBATCH --job-name=osei-stage0-loader
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/slurm-osei-stage0-loader-%j.out
#SBATCH --error=results/slurm-osei-stage0-loader-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

mkdir -p results/stage0

JOB_ID="${SLURM_JOB_ID:-manual}"
REPORT_PATH="results/stage0/stage0_strict_loader_audit_${JOB_ID}.json"

# 使用项目本地虚拟环境，避免系统 Python 依赖漂移影响审计结论。
.venv/bin/python tools/stage0_strict_loader_audit.py \
    --project-root "${PROJECT_ROOT}" \
    --data-root "数据集" \
    --dataset all \
    --output "${REPORT_PATH}"

echo "阶段 0 strict loader 审计报告：${PROJECT_ROOT}/${REPORT_PATH}"
