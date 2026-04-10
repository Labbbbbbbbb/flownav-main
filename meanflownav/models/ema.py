"""
MeanFlow EMA (Exponential Moving Average).

Reuses the EMA utilities from py-meanflow directly.
- init_ema(net, net_ema, ema_decay): copy weights, disable grad
- update_ema_net(net, net_ema, num_updates, period=16): periodic update with double precision
"""

from meanflow.models.ema import init_ema, update_ema_net

__all__ = ["init_ema", "update_ema_net"]
