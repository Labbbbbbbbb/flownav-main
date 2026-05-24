#!/bin/bash
# MeanFlow Navigation Training Script

# ---- Config ----
CONDA_ENV="flownav"
CONFIG="meanflownav/config/meanflownav.yaml"


# ---- Activate environment ----
# eval "$(conda shell.bash hook)"
# conda activate ${CONDA_ENV}
# source $(pwd)/.venv/bin/activate

# ---- Add project paths ----
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/consistency-policy:$(pwd)/py-meanflow"

# ---- Run ----
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_meanflow.py --config ${CONFIG}
