#!/usr/bin/env bash
#SBATCH --job-name=robomme-train-1k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:6
#SBATCH --mem=256G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/post_train/slurm-%x-%j.out
#SBATCH --error=logs/post_train/slurm-%x-%j.err

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lingbotva
cd "${SLURM_SUBMIT_DIR}"

NGPU=${NGPU:-6}
MASTER_PORT=${MASTER_PORT:-29501}
LOG_RANK=${LOG_RANK:-0}
ROBOMME_SAVE_ROOT=${ROBOMME_SAVE_ROOT:-/DATA/disk0/yry/LingbotVa/results/lingbotva_robomme_1k}

export ROBOMME_LEROBOT_DATASET_PATH=/DATA/disk0/yry/robomme_data_lerobot
export LINGBOT_ROBOMME_ACTION_STATS=${ROBOMME_LEROBOT_DATASET_PATH}/robomme_action_stats.json

# 6 卡、1,000 optimizer steps、global batch 36。
export ROBOMME_ENABLE_WANDB=${ROBOMME_ENABLE_WANDB:-1}
export ROBOMME_LOAD_WORKER=${ROBOMME_LOAD_WORKER:-4}
export ROBOMME_BATCH_SIZE=${ROBOMME_BATCH_SIZE:-1}
export ROBOMME_GRADIENT_ACCUMULATION_STEPS=${ROBOMME_GRADIENT_ACCUMULATION_STEPS:-6}
export ROBOMME_NUM_STEPS=${ROBOMME_NUM_STEPS:-1000}
export ROBOMME_SAVE_INTERVAL=${ROBOMME_SAVE_INTERVAL:-200}
# Leave empty for a new run. To resume, export a checkpoint directory such as:
# ROBOMME_RESUME_FROM=/DATA/disk0/yry/LingbotVa/results/lingbotva_robomme_1k/checkpoints/checkpoint_step_600
export ROBOMME_RESUME_FROM="${ROBOMME_RESUME_FROM:-}"

export WANDB_TEAM_NAME=${WANDB_TEAM_NAME:-yuanruyi-chonqinguniversity}
export WANDB_PROJECT=${WANDB_PROJECT:-lingbotva-posttraining-robomme}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_NAME=${WANDB_NAME:-robomme-1k-${SLURM_JOB_ID}}
export WANDB_DIR=${WANDB_DIR:-/DATA/disk0/yry/LingbotVa/results}
mkdir -p "${WANDB_DIR}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "启动 6 卡、1,000 optimizer steps RoboMME 后训练"
srun python -m torch.distributed.run \
    --nproc_per_node="${NGPU}" \
    --local-ranks-filter="${LOG_RANK}" \
    --master_port="${MASTER_PORT}" \
    --tee=3 \
    -m wan_va.train \
    --config-name robomme_train \
    --save-root "${ROBOMME_SAVE_ROOT}"
