#!/bin/bash
# 阶段 3 WiSig RADCIL high-replay 正式三种子后端消融。
#
# 用途：
#   1. 复用阶段 3 主实验入口和严格 WiSig 10+10x3 协议；
#   2. 仅把后端 old:new replay batch ratio 从 2.0 提高到 3.0；
#   3. 运行 seed 7/13/31，生成可与主方法直接对比的正式消融报告。
#
# 分区可在提交时覆盖，例如：
#   sbatch --partition=gpu --qos=normal --time=04:00:00 slurm/stage3_wisig_high_replay_multiseed.sbatch

#SBATCH --job-name=osei-stage3-highreplay
#SBATCH --partition=gpuHz
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=shortjobs
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=results/slurm-osei-stage3-highreplay-%j.out
#SBATCH --error=results/slurm-osei-stage3-highreplay-%j.err

set -euo pipefail

# 通过环境变量复用主实验脚本，确保 high-replay 消融和主方法共享同一套
# 计划生成、训练命令和报告汇总逻辑；这里不复制实验参数，避免后续漂移。
export RADCIL_STAGE3_VARIANT="ratio_3p0_replay_3p0"
export RADCIL_STAGE3_OUTPUT_PREFIX="${RADCIL_STAGE3_OUTPUT_PREFIX:-results/stage3/wisig_high_replay}"

bash slurm/stage3_wisig_main_multiseed.sbatch
