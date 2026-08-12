#!/usr/bin/env bash
#SBATCH --job-name=robomme-vae-1000-1199
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lingbotva
cd /DATA/disk0/yry/LingbotVa

srun python wan_va/utils/extract_robomme_latents.py \
  --dataset-root /DATA/disk0/yry/robomme_data_lerobot \
  --model-path /DATA/disk0/yry/lingbot-va-base \
  --camera-keys image wrist_image \
  --height 256 \
  --width 256 \
  --target-fps 10 \
  --max-sequence-length 512 \
  --dtype bfloat16 \
  --device cuda \
  --text-encoder-device cpu \
  --episode-start 1000 \
  --episode-end 1200 \
  --force
