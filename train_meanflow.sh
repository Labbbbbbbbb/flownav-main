#!/bin/bash
# MeanFlow Navigation Training Script

# ---- Config ----
CONDA_ENV="flownav"
CONFIG="meanflownav/config/meanflownav.yaml"
GPU_ID=0

# ---- Activate environment ----
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

# ---- Add project paths ----
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/consistency-policy:$(pwd)/py-meanflow"

# ---- Run ----
CUDA_VISIBLE_DEVICES=${GPU_ID} python train_meanflow.py --config ${CONFIG}
