#!/bin/bash
# Stage 21：ADS-B 联合 discovery-CIL 三种子确认。
#
# 参数固定为已经通过 seed7 验证的联合闭环配置：
# GPCC -> 第一次 CIL -> Student 重新发现 -> 无标签 Hungarian 对齐 -> 第二次 CIL。
# 该任务只验证 seed 7/13/31 的稳定性，不在运行中搜索相邻权重。

#SBATCH --job-name=osei-s21-adsb-ms
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage21/slurm_adsb_joint_discovery_cil_multiseed_%j.out
#SBATCH --error=results/stage21/slurm_adsb_joint_discovery_cil_multiseed_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
ADSB_ROOT="${ADSB_DATA_ROOT:-${DATA_ROOT}/数据集/ADS-B/Dataset}"
SEEDS="${STAGE21_ADSB_SEEDS:-7,13,31}"

mkdir -p results/stage21
IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"

for seed in "${SEED_ARRAY[@]}"; do
    seed="$(echo "${seed}" | tr -d '[:space:]')"
    SAVE_DIR="results/stage21/adsb_joint_discovery_cil_seed${seed}_${JOB_ID}"
    CHECKPOINT="${SAVE_DIR}/closedset_adsb_90known_strict.pth"
    mkdir -p "${SAVE_DIR}"

    echo "运行 Stage 21 联合 discovery-CIL：seed=${seed}"
    "${PYTHON_BIN}" experiments/exp_adsb_mvacc_cil_strict.py \
        --data_root "${ADSB_ROOT}" \
        --save_dir "${SAVE_DIR}" \
        --checkpoint "${CHECKPOINT}" \
        --train_closedset \
        --epochs 12 \
        --initial_known_classes 90 --round_size 10 --num_rounds 3 --seed "${seed}" \
        --backbone adsb_long --batch_size 128 --test_batch_size 256 \
        --use_supcon --supcon_weight 0.30 \
        --discovery_backend gpcc \
        --enable_joint_discovery_refinement \
        --disable_cil_baselines --disable_visualization \
        --cil_head_warmup_epochs 2 \
        --cil_joint_epochs 8 \
        --cil_replay_weight 3.0 \
        --radcil_old_new_batch_ratio 2.0
done

"${PYTHON_BIN}" tools/stage21_adsb_joint_discovery_multiseed_report.py \
    --root results/stage21 \
    --job-id "${JOB_ID}" \
    --seeds "${SEEDS}" \
    --output "results/stage21/STAGE21_ADSB_JOINT_DISCOVERY_MULTISEED_REPORT_${JOB_ID}.md" \
    --summary-json "results/stage21/stage21_adsb_joint_discovery_multiseed_summary_${JOB_ID}.json"

echo "Stage 21 ADS-B 联合 discovery-CIL 三种子实验完成。"
