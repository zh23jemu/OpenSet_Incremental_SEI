#!/bin/bash
# Stage 59：LoRa 物理描述符发现 + 原型分类 seed7 快速验证。
#
# 背景：
#   Stage53-58 连续证明 RADCIL/head/后验路由类机制会陷入 Old/New 跷跷板。
#   本阶段完全绕开神经增量头，直接用 LoRa I/Q 的幅度、相位差和频谱
#   物理描述符做固定 K=5 discovery 与 nearest-prototype 识别。
#
# 验证内容：
#   kmeans       : 物理描述符 + KMeans(K=5) + 原型分类；
#   kmeans_rec065: 同上，但对 discovery 聚类做 recording consensus 0.65。
#
# 通过门槛：
#   相对 Stage48 三种子，R3 Overall 至少 +0.01；Old 不明显下降；
#   New 下降不超过 0.05。否则不扩三种子。
#SBATCH --job-name=stage59_lora_phys_proto
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=results/stage59/slurm_lora_physical_descriptor_proto_seed7_%j.out
#SBATCH --error=results/stage59/slurm_lora_physical_descriptor_proto_seed7_%j.err

set -euo pipefail

module load slurm

DATA_ROOT="${DATA_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python"
fi
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 127
fi

JOB_ID="${SLURM_JOB_ID:-manual}"
ROOT="results/stage59"
S28_NPZ="${LORA25_S28_NPZ:-results/stage46/lora25_diffdays_indoor_raw1024_s28_46103936.npz}"
mkdir -p "${ROOT}"

echo "Stage59 data_root=${DATA_ROOT}"
echo "Stage59 s28_npz=${S28_NPZ}"
echo "Stage59 python=${PYTHON_BIN}"
if [ ! -f "${S28_NPZ}" ]; then
    echo "Missing s28 NPZ: ${S28_NPZ}. Run Stage46 first or pass LORA25_S28_NPZ." >&2
    exit 2
fi

run_variant() {
    local variant="$1"
    local consensus="$2"
    local save_dir="${ROOT}/lora_s59_${variant}_${JOB_ID}"
    mkdir -p "${save_dir}"
    echo "== Stage59 variant ${variant}: consensus=${consensus}, save_dir=${save_dir} =="
    "${PYTHON_BIN}" tools/stage59_lora_physical_descriptor_proto.py \
        --dataset-path "${S28_NPZ}" \
        --save-dir "${save_dir}" \
        --seed 7 \
        --pca-dim 64 \
        --spectrum-bins 32 \
        --recording-consensus-threshold "${consensus}"
}

run_variant kmeans 0.0
run_variant kmeans_rec065 0.65
