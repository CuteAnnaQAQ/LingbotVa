#!/usr/bin/env bash
#SBATCH --job-name=robomme-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lingbotva
cd "${SLURM_SUBMIT_DIR}"

export ROBOMME_LEROBOT_DATASET_PATH=/DATA/disk0/yry/robomme_data_lerobot
export LINGBOT_ROBOMME_ACTION_STATS=${ROBOMME_LEROBOT_DATASET_PATH}/robomme_action_stats.json

# 4 卡、20 step 的首次连通性测试。
export ROBOMME_ENABLE_WANDB=0
export ROBOMME_LOAD_WORKER=4
export ROBOMME_BATCH_SIZE=1
export ROBOMME_GRADIENT_ACCUMULATION_STEPS=1
export ROBOMME_NUM_STEPS=20
export ROBOMME_SAVE_INTERVAL=10

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "检查真实 dataset loader 单样本"
srun python -m wan_va.utils.inspect_robomme_sample

echo "启动 4 卡、20 step smoke test"
srun python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29501 \
    --tee=3 \
    -m wan_va.train \
    --config-name robomme_train \
    --save-root /DATA/disk0/yry/lingbotva_robomme_smoke
