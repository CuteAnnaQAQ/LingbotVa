#!/usr/bin/env bash
#SBATCH --job-name=robomme-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:6
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=logs/post_train/slurm-%x-%j.out
#SBATCH --error=logs/post_train/slurm-%x-%j.err

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lingbotva
cd "${SLURM_SUBMIT_DIR}"

NGPU=${NGPU:-6}
MASTER_PORT=${MASTER_PORT:-29501}
LOG_RANK=${LOG_RANK:-0}

export ROBOMME_LEROBOT_DATASET_PATH=/DATA/disk0/yry/robomme_data_lerobot
export LINGBOT_ROBOMME_ACTION_STATS=${ROBOMME_LEROBOT_DATASET_PATH}/robomme_action_stats.json

# 6 卡、20 step 的首次连通性测试。
export ROBOMME_ENABLE_WANDB=0
export ROBOMME_LOAD_WORKER=4
export ROBOMME_BATCH_SIZE=1
export ROBOMME_GRADIENT_ACCUMULATION_STEPS=1
export ROBOMME_NUM_STEPS=20
export ROBOMME_SAVE_INTERVAL=10

export WANDB_API_KEY="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"
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}
export WANDB_TEAM_NAME=${WANDB_TEAM_NAME:-yuanruyi-chonqinguniversity}
export WANDB_PROJECT=${WANDB_PROJECT:-lingbotva-posttraining-robomme}

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "启动 6 卡、20 step smoke test"
srun python -m torch.distributed.run \
    --nproc_per_node="${NGPU}" \
    --local-ranks-filter="${LOG_RANK}" \
    --master_port="${MASTER_PORT}" \
    --tee=3 \
    -m wan_va.train \
    --config-name robomme_train \
    --save-root /DATA/disk0/yry/LingbotVa/results/lingbotva_robomme_smoke
