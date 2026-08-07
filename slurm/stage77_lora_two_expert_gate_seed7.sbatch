#!/usr/bin/env bash
# Stage77：分离旧类跨天专家与当前伪新类增量头。
# 本任务只做训练后门控验证，复用 Stage76 已生成模型；不重跑完整 CIL。
# IQ_8-10 held-out 标签只用于脚本内部最终指标，绝不进入专家、门控和阈值选择。

#SBATCH --job-name=stage77-lora-expert-gate
#SBATCH --partition=gpu
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage77/slurm_lora_two_expert_gate_seed7_%j.out
#SBATCH --error=results/stage77/slurm_lora_two_expert_gate_seed7_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI_stage75}"
DATA_ROOT="${DATA_ROOT:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI}"
PYTHON_BIN="${PYTHON_BIN:-${DATA_ROOT}/.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    # 独立 Git worktree 不复制虚拟环境，回退到主项目已经验证过的解释器。
    PYTHON_BIN="/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python"
fi
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 127
fi
RAW_DIR="${LORA_RAW_DIR:-/mnt/usmidet/billy_test/OpenSet_Incremental_SEI/stage44/raw_setup1_iq}"
STAGE76_DIR="${STAGE76_DIR:-${PROJECT_ROOT}/results/stage76/lora_old_day_joint_cil_seed7_46630851}"
DATASET_PATH="${LORA_S28_NPZ:-${PROJECT_ROOT}/results/stage76/lora25_diffdays_indoor_raw1024_s28_46630851.npz}"
OUTPUT_DIR="${PROJECT_ROOT}/results/stage77/lora_two_expert_gate_seed7_${SLURM_JOB_ID}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" tools/stage77_lora_old_new_expert_gate.py \
    --dataset-path "${DATASET_PATH}" \
    --raw-dir "${RAW_DIR}" \
    --stage76-save-dir "${STAGE76_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --transmissions 1,2,3,4,5,6,7 \
    --symbols-per-transmission 28 \
    --decimation 1 \
    --seed 7 \
    --device cuda
