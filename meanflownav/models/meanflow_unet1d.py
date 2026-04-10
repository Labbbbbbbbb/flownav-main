"""
MeanFlow UNet1D for action trajectory generation.

Reuses CTMConditionalUnet1D from consistency-policy, which already supports
dual timestep input (timestep, stoptime) — mapping directly to MeanFlow's (t, h).

Usage:
    model = MeanFlowConditionalUnet1D(
        input_dim=2,
        global_cond_dim=256,
        down_dims=[64, 128, 256],
    )
    # t: current timestep, h = t - r: time difference
    output = model(sample=z, timestep=t, stoptime=h, global_cond=obs_cond)
"""

import sys
import os

# Add consistency-policy to path for imports
_consistency_policy_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "consistency-policy"
)
if os.path.isdir(_consistency_policy_path):
    sys.path.insert(0, os.path.abspath(_consistency_policy_path))

from consistency_policy.ctm_unet import CTMConditionalUnet1D as MeanFlowConditionalUnet1D

__all__ = ["MeanFlowConditionalUnet1D"]
