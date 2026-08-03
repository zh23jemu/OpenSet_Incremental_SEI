#!/bin/bash
# Stage 44：LoRa Setup 1 原始 I/Q 必要文件准备入口。
#
# 默认只生成 manifest 并探测少量远端链接，不会下载完整大数据。
# 若确认磁盘和网络都可用，可提交时显式打开：
#   DOWNLOAD_RAW=1 CHECK_REMOTE=1 sbatch --partition=defq --qos=normal slurm/stage44_lora_setup1_raw_iq_prepare.sbatch
#
# 可选环境变量：
#   RAW_DIR       原始 I/Q 保存目录，默认 datasets/lora25_compact/raw_setup1_iq
#   LIMIT         只处理前 N 条记录；空值表示全量必要清单
#   DOWNLOAD_RAW  设为 1 时实际下载必要 IQ_*.dat
#   CHECK_REMOTE  设为 1 时探测远端文件大小

#SBATCH --job-name=osei-s44-lora-rawiq
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=results/stage44/slurm_lora_setup1_raw_iq_prepare_%j.out
#SBATCH --error=results/stage44/slurm_lora_setup1_raw_iq_prepare_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-local}"
RAW_DIR="${RAW_DIR:-datasets/lora25_compact/raw_setup1_iq}"
MANIFEST="results/stage44/lora_setup1_raw_iq_manifest_${JOB_ID}.json"

cd "${PROJECT_ROOT}"
mkdir -p results/stage44

ARGS=(
    datasets/lora25_compact/tools/download_lora25_setup1_raw_iq.py
    --raw-dir "${RAW_DIR}"
    --manifest "${MANIFEST}"
)

if [[ "${CHECK_REMOTE:-1}" == "1" ]]; then
    ARGS+=(--check-remote)
fi

if [[ "${DOWNLOAD_RAW:-0}" == "1" ]]; then
    ARGS+=(--download)
fi

if [[ -n "${LIMIT:-}" ]]; then
    ARGS+=(--limit "${LIMIT}")
fi

echo "运行 LoRa Setup 1 原始 I/Q 准备：raw_dir=${RAW_DIR}, manifest=${MANIFEST}, download=${DOWNLOAD_RAW:-0}, check_remote=${CHECK_REMOTE:-1}, limit=${LIMIT:-all}"
"${PYTHON_BIN}" "${ARGS[@]}"
echo "LoRa Setup 1 原始 I/Q 准备完成：${PROJECT_ROOT}/${MANIFEST}"
