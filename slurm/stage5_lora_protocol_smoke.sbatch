#!/bin/bash
# 阶段 5：LoRa25 strict 协议 Slurm smoke test。
#
# 用途：
#   1. 在 Slurm 项目环境中复核 LoRa RFFP Different Days Indoor 紧凑 NPZ；
#   2. 固化 10 known + 5x3 的跨天开放集增量边界；
#   3. 只生成协议审计 JSON，不训练模型、不读取 held-out eval 真值做选参。
#
# 分区可在提交时覆盖，例如：
#   sbatch --partition=defq --qos=shortjobs --time=00:30:00 slurm/stage5_lora_protocol_smoke.sbatch

#SBATCH --job-name=osei-stage5-lora-audit
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=results/slurm-osei-stage5-lora-audit-%j.out
#SBATCH --error=results/slurm-osei-stage5-lora-audit-%j.err

set -euo pipefail

module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

mkdir -p results/stage5

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
JOB_ID="${SLURM_JOB_ID:-manual}"
LORA_NPZ="${LORA25_NPZ:-datasets/lora25_compact/lora25_diffdays_indoor_aligned_group_256.npz}"
OUTPUT_PATH="results/stage5/stage5_lora_protocol_audit_${JOB_ID}.json"

echo "运行阶段 5 LoRa25 strict 协议审计：npz=${LORA_NPZ}, output=${OUTPUT_PATH}"
"${PYTHON_BIN}" tools/stage5_lora_protocol_audit.py \
    --npz-path "${LORA_NPZ}" \
    --source-manifest datasets/lora25_compact/lora25_source_manifest.json \
    --alignment-manifest datasets/lora25_compact/lora25_alignment_group_manifest_256.json \
    --output "${OUTPUT_PATH}"

echo "阶段 5 LoRa25 strict 协议审计完成：${PROJECT_ROOT}/${OUTPUT_PATH}"
