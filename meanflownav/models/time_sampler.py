"""
MeanFlow dual-timestep (t, r) sampler.

Reuses the time_sampler from py-meanflow directly.
Requires an `args` namespace with attributes:
    tr_sampler, ratio, P_mean_t, P_std_t, P_mean_r, P_std_r
"""

from meanflow.models.time_sampler import (
    logit_normal_timestep_sample,
    sample_two_timesteps,
)

__all__ = ["logit_normal_timestep_sample", "sample_two_timesteps"]
