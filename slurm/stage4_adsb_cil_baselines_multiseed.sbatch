#!/bin/bash
# ADS-B strict 核心 CIL baseline 与发现前端三种子对照。
#
# 复用正式长序列 checkpoint，不重新训练闭集模型；不传
# --disable_cil_baselines，从而运行共享 MV-ACC 发现后端和 Deep-HDBSCAN
# 端到端两组 Ft-CNN/LwF/iCaRL/EEIL/TPCIL-style/DOI-style 对照。
# 可在提交时用 sbatch --partition=目标分区 覆盖默认 gpu 分区。

#SBATCH --job-name=osei-stage4-adsb-baselines
#SBATCH --partition=gpu
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage4-adsb-baselines-%j.out
#SBATCH --error=results/slurm-osei-stage4-adsb-baselines-%j.err

set -euo pipefail
module load slurm

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/users/xj62kv/OpenSet_Incremental_SEI}"
cd "${PROJECT_ROOT}"
mkdir -p results/stage4

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DATA_ROOT="${ADSB_DATA_ROOT:-数据集/ADS-B/Dataset}"
JOB_ID="${SLURM_JOB_ID:-manual}"
SEEDS="${ADSB_STAGE4_BASELINE_SEEDS:-7,13,31}"
OUTPUT_PREFIX="${ADSB_STAGE4_BASELINE_OUTPUT_PREFIX:-results/stage4/adsb_cil_baselines}"

checkpoint_for_seed() {
    case "$1" in
        7) echo "results/stage4/adsb_main_long_radcil_seed7_44470736/closedset_adsb_90known_strict.pth" ;;
        13) echo "results/stage4/adsb_main_long_radcil_seed13_44470736/closedset_adsb_90known_strict.pth" ;;
        31) echo "results/stage4/adsb_main_long_radcil_seed31_44467424/closedset_adsb_90known_strict.pth" ;;
        *) echo "不支持的 seed：$1" >&2; return 1 ;;
    esac
}

IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
for seed in "${SEED_ARRAY[@]}"; do
    seed="$(echo "${seed}" | tr -d '[:space:]')"
    [[ -z "${seed}" ]] && continue
    checkpoint="$(checkpoint_for_seed "${seed}")"
    save_dir="${OUTPUT_PREFIX}_seed${seed}_${JOB_ID}"
    echo "运行 ADS-B strict baseline：seed=${seed}, save_dir=${save_dir}"
    "${PYTHON_BIN}" experiments/exp_adsb_mvacc_cil_strict.py \
        --data_root "${DATA_ROOT}" \
        --save_dir "${save_dir}" \
        --checkpoint "${checkpoint}" \
        --initial_known_classes 90 \
        --round_size 10 \
        --num_rounds 3 \
        --seed "${seed}" \
        --backbone adsb_long \
        --batch_size 128 \
        --test_batch_size 256 \
        --use_supcon \
        --supcon_weight 0.1 \
        --cil_head_warmup_epochs 2 \
        --cil_joint_epochs 8 \
        --cil_replay_weight 3.0 \
        --radcil_old_new_batch_ratio 2.0 \
        --disable_mvacc_calibration \
        --mvacc_min_cluster_ratio 0.03 \
        --mvacc_merge_threshold 0.74 \
        --mvacc_split_size_factor 1.75 \
        --mvacc_split_silhouette 0.30 \
        --disable_visualization
done

"${PYTHON_BIN}" tools/stage4_adsb_cil_baseline_report.py \
    --job-id "${JOB_ID}" \
    --seeds "${SEEDS}" \
    --output-prefix "${OUTPUT_PREFIX}" \
    --output "results/stage4/STAGE4_ADSB_STRICT_BASELINE_TABLE_${JOB_ID}.md" \
    --summary-json "results/stage4/stage4_adsb_cil_baselines_summary_${JOB_ID}.json"

echo "ADS-B strict baseline 三种子完成：results/stage4/STAGE4_ADSB_STRICT_BASELINE_TABLE_${JOB_ID}.md"
