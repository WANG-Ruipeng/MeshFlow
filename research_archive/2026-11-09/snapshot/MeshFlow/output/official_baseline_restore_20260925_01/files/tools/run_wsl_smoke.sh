#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
MESHFlow_ENV="${MESHFLOW_ENV:-$HOME/miniforge3/envs/meshflow}"
export PATH="$MESHFlow_ENV/bin:$PATH"
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
accelerate launch \
  --num_processes 1 --num_machines 1 \
  --mixed_precision bf16 --dynamo_backend no --gpu_ids 0 \
  train.py --config configs/local/wsl5080-smoke.yaml "$@"
