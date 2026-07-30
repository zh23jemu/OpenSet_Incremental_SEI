#!/bin/bash
# Stage 13：ADS-B cross_day + GPCC + RADCIL 三种子确认。
#
# 只扩展 ADS-B，因为 Stage 12 seed7 已达到继续验证门槛；LoRa 当前仍有
# 明显旧新类失衡，不在本作业中重复扩展。

#SBATCH --job-name=osei-s13-adsb-cross-day
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=results/stage13/slurm_adsb_cross_day_multiseed_%j.out
#SBATCH --error=results/stage13/slurm_adsb_cross_day_multiseed_%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATA_ROOT="${DATA_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/users/xj62kv/OpenSet_Incremental_SEI/.venv/bin/python}"
JOB_ID="${SLURM_JOB_ID:-manual}"
ADSB_ROOT="${ADSB_DATA_ROOT:-${DATA_ROOT}/数据集/ADS-B/Dataset}"
mkdir -p results/stage13

checkpoint_for_seed() {
    case "$1" in
        7) echo "${DATA_ROOT}/results/stage4/adsb_main_long_radcil_seed7_44470736/closedset_adsb_90known_strict.pth" ;;
        13) echo "${DATA_ROOT}/results/stage4/adsb_main_long_radcil_seed13_44470736/closedset_adsb_90known_strict.pth" ;;
        31) echo "${DATA_ROOT}/results/stage4/adsb_main_long_radcil_seed31_44467424/closedset_adsb_90known_strict.pth" ;;
        *) return 1 ;;
    esac
}

for seed in 7 13 31; do
    CHECKPOINT="$(checkpoint_for_seed "${seed}")"
    SAVE_DIR="results/stage13/adsb_cross_day_seed${seed}_${JOB_ID}"
    "${PYTHON_BIN}" experiments/exp_adsb_mvacc_cil_strict.py \
        --data_root "${ADSB_ROOT}" \
        --save_dir "${SAVE_DIR}" \
        --checkpoint "${CHECKPOINT}" \
        --initial_known_classes 90 --round_size 10 --num_rounds 3 --seed "${seed}" \
        --backbone adsb_long --batch_size 128 --test_batch_size 256 \
        --use_supcon --supcon_weight 0.1 \
        --disable_cil_baselines --disable_visualization \
        --discovery_backend gpcc \
        --cross_day_repr_adaptation \
        --cross_day_repr_epochs 2 \
        --cross_day_repr_batch_size 128 \
        --cross_day_repr_lr 1e-5 \
        --cross_day_repr_consistency_weight 0.5 \
        --cross_day_repr_coral_weight 0.05 \
        --cil_head_warmup_epochs 2 \
        --cil_joint_epochs 8 \
        --cil_replay_weight 3.0 \
        --radcil_old_new_batch_ratio 2.0
done

"${PYTHON_BIN}" tools/stage13_adsb_cross_day_multiseed_report.py \
    --root results/stage13 \
    --job-id "${JOB_ID}" \
    --output "results/stage13/STAGE13_ADSB_CROSS_DAY_MULTISEED_REPORT_${JOB_ID}.md"
