#!/bin/bash
# FlowCorrect Offline Training Script

# ---- Config ----
CONDA_ENV="flownav"
CONFIG="meanflownav/config/flow_correct.yaml"
GPU_ID=6

# ---- Activate environment ----
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

# ---- Add project paths ----
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/thirdparty/consistency-policy:$(pwd)/thirdparty/py-meanflow"

# ---- Run ----
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_flow_correct.py --config ${CONFIG}
