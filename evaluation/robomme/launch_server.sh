#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

START_PORT="${START_PORT:-29536}"
MASTER_PORT="${MASTER_PORT:-29501}"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/outputs/robomme_smoke_server}"

mkdir -p "${SAVE_ROOT}"
cd "${REPO_ROOT}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port "${MASTER_PORT}" \
    -m wan_va.wan_va_server \
    --config-name robomme \
    --port "${START_PORT}" \
    --save_root "${SAVE_ROOT}"
