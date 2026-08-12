#!/usr/bin/env bash
#SBATCH --job-name=robomme-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lingbotva

cd "${SLURM_SUBMIT_DIR}"

DATASET_ROOT=/DATA/disk0/yry/robomme_data_lerobot
ACTION_STATS=${DATASET_ROOT}/robomme_action_stats.json
EXTRACT_ARGS=(
    --dataset-root "${DATASET_ROOT}"
    --max-sequence-length 512
)
if [[ "${ROBOMME_FORCE_REEXTRACT:-0}" == "1" ]]; then
    EXTRACT_ARGS+=(--force)
fi

echo "[1/3] 提取 latent"
srun python wan_va/utils/extract_robomme_latents.py "${EXTRACT_ARGS[@]}"

echo "[2/3] 计算 action 分位数"
srun python -m wan_va.utils.compute_robomme_action_stats \
    "${DATASET_ROOT}" --output "${ACTION_STATS}"

echo "[3/3] 校验 LingBot-ready"
srun python -m wan_va.utils.validate_robomme_lerobot \
    --dataset-root "${DATASET_ROOT}" \
    --action-stats "${ACTION_STATS}" \
    --scan-rows --inspect-latents --require-lingbot-ready
