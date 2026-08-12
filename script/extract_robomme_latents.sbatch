#!/usr/bin/env bash
#SBATCH --job-name=robomme-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

umask 007

# 默认从 lingbot-va/ 提交；也允许从其父目录提交，或用 LINGBOT_VA_ROOT 显式指定。
submit_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
repo_root="${LINGBOT_VA_ROOT:-${submit_dir}}"
if [[ ! -f "${repo_root}/wan_va/utils/extract_robomme_latents.py" ]]; then
    if [[ -f "${repo_root}/lingbot-va/wan_va/utils/extract_robomme_latents.py" ]]; then
        repo_root="${repo_root}/lingbot-va"
    else
        echo "无法定位 lingbot-va；请从仓库目录提交，或设置 LINGBOT_VA_ROOT。" >&2
        exit 1
    fi
fi
cd "${repo_root}"

# sbatch 默认继承提交时的环境；需要使用指定解释器时可设置 PYTHON_BIN。
python_bin="${PYTHON_BIN:-python}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "找不到 Python：${python_bin}；请先激活 LingBot 环境再提交。" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# 三个阶段必须使用同一个数据集根目录。环境变量可覆盖这里的默认路径。
dataset_root="${ROBOMME_LEROBOT_DATASET_PATH:-/DATA/disk0/yry/robomme_data_lerobot}"
action_stats_path="${LINGBOT_ROBOMME_ACTION_STATS:-${dataset_root}/robomme_action_stats.json}"
export ROBOMME_LEROBOT_DATASET_PATH="${dataset_root}"
export LINGBOT_ROBOMME_ACTION_STATS="${action_stats_path}"

echo "job_id=${SLURM_JOB_ID:-local}"
echo "host=$(hostname)"
echo "repo_root=${repo_root}"
echo "python=$(command -v "${python_bin}")"
echo "dataset_root=${dataset_root}"
echo "action_stats_path=${action_stats_path}"
nvidia-smi || true

# 数据集路径由流水线统一传入；其余 sbatch 参数原样转发给 extractor。
echo "[1/3] 提取 image 和 wrist_image 的 Wan2.2 latent"
srun "${python_bin}" wan_va/utils/extract_robomme_latents.py \
    "$@" \
    --dataset-root "${dataset_root}"

# dry-run 只规划提取任务，不应继续生成统计量或执行必然失败的完整性校验。
for argument in "$@"; do
    if [[ "${argument}" == "--dry-run" ]]; then
        echo "dry-run 完成；跳过 action 统计和 LingBot-ready 校验。"
        exit 0
    fi
done

echo "[2/3] 计算官方 absolute_joint actions 的 q01/q99"
srun "${python_bin}" wan_va/utils/compute_robomme_action_stats.py \
    "${dataset_root}" \
    --action-key actions \
    --output "${action_stats_path}"

echo "[3/3] 执行严格 LingBot-ready 校验"
srun "${python_bin}" wan_va/utils/validate_robomme_lerobot.py \
    --dataset-root "${dataset_root}" \
    --action-stats "${action_stats_path}" \
    --scan-rows \
    --inspect-latents \
    --require-lingbot-ready

echo "RoboMME 后训练数据准备完成：latent、action 分位数和严格校验均已通过。"
