#!/bin/bash
# LoRa25 seed7 跨 transmission 表征筛选；只用 IQ_7 选择候选。
#SBATCH --job-name=osei-stage5-lora-repr-s7
#SBATCH --partition=defq
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=results/slurm-osei-stage5-lora-repr-s7-%j.out
#SBATCH --error=results/slurm-osei-stage5-lora-repr-s7-%j.err

set -euo pipefail
module load slurm
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
JOB_ID="${SLURM_JOB_ID:-manual}"
"${PROJECT_ROOT}/.venv/bin/python" tools/stage5_lora_representation_screen.py \
    --seed 7 \
    --epochs 20 \
    --output-dir "results/stage5/lora_representation_screen_seed7_${JOB_ID}"
