"""
MeanFlow inference utilities — one-step generation replaces 10-step ODE.

Reuses data utilities from flownav.training.utils (normalize, get_action, etc.).
Only redefines model_output() for single-step MeanFlow inference.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

# Reuse all data utilities from original flownav
from flownav.training.utils import (
    ACTION_STATS,
    action_reduce,
    get_action,
    to_numpy,
    from_numpy,
    normalize_data,
    unnormalize_data,
    get_delta,
    load_data_stats,
    visualize_action_distribution as _original_visualize,
)


def model_output(
    model: nn.Module,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
    use_wandb: bool,
) -> dict[str, torch.Tensor]:
    """One-step MeanFlow inference, replacing 10-step ODE integration."""

    # Unwrap DataParallel if needed
    model_unwrapped = model.module if hasattr(model, "module") else model

    # Exploration (goal masked)
    goal_mask = torch.ones((batch_goal_images.shape[0],)).long().to(device)
    obs_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_img=batch_goal_images,
        input_goal_mask=goal_mask,
    )
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

    # Navigation (no mask)
    no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
    obsgoal_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_img=batch_goal_images,
        input_goal_mask=no_mask,
    )
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    with torch.no_grad():
        start_time = time.time()

        # Exploration — one-step MeanFlow
        e = torch.randn(
            (len(obs_cond), pred_horizon, action_dim), device=device
        )
        t = torch.ones(e.shape[0], device=device)
        h = torch.ones(e.shape[0], device=device)
        u = model_unwrapped.noise_pred_net(
            sample=e, timestep=t, stoptime=h, global_cond=obs_cond
        )
        uc_actions = get_action(e - u, ACTION_STATS)

        proc_time = time.time() - start_time
        if use_wandb:
            wandb.log({"Mean Processing Time UC": proc_time / e.shape[0]})
            wandb.log({"Processing Time UC": proc_time})

        # Navigation — one-step MeanFlow
        e = torch.randn(
            (len(obs_cond), pred_horizon, action_dim), device=device
        )
        u = model_unwrapped.noise_pred_net(
            sample=e, timestep=t, stoptime=h, global_cond=obsgoal_cond
        )
        gc_actions = get_action(e - u, ACTION_STATS)

        proc_time = time.time() - start_time
        if use_wandb:
            wandb.log({"Mean Processing Time GC": proc_time / e.shape[0]})
            wandb.log({"Processing Time GC": proc_time})

    # Predict distance
    obsgoal_cond_flat = obsgoal_cond.flatten(start_dim=1)
    gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond_flat)

    return {
        "uc_actions": uc_actions,
        "gc_actions": gc_actions,
        "gc_distance": gc_distance,
    }


def compute_losses(
    ema_model: nn.Module,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_dist_label: torch.Tensor,
    batch_action_label: torch.Tensor,
    device: torch.device,
    action_mask: torch.Tensor,
    use_wandb: bool,
) -> dict[str, torch.Tensor]:
    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    output_dict = model_output(
        model=ema_model,
        batch_obs_images=batch_obs_images,
        batch_goal_images=batch_goal_images,
        pred_horizon=pred_horizon,
        action_dim=action_dim,
        num_samples=1,
        device=device,
        use_wandb=use_wandb,
    )
    uc_actions = output_dict["uc_actions"]
    gc_actions = output_dict["gc_actions"]
    gc_distance = output_dict["gc_distance"]

    gc_dist_loss = F.mse_loss(gc_distance, batch_dist_label.unsqueeze(-1))

    uc_action_loss = action_reduce(
        F.mse_loss(uc_actions, batch_action_label, reduction="none"),
        action_mask,
    )
    gc_action_loss = action_reduce(
        F.mse_loss(gc_actions, batch_action_label, reduction="none"),
        action_mask,
    )

    uc_action_waypts_cos_sim = action_reduce(
        F.cosine_similarity(
            uc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
        ),
        action_mask,
    )
    uc_multi_action_waypts_cos_sim = action_reduce(
        F.cosine_similarity(
            torch.flatten(uc_actions[:, :, :2], start_dim=1),
            torch.flatten(batch_action_label[:, :, :2], start_dim=1),
            dim=-1,
        ),
        action_mask,
    )
    gc_action_waypts_cos_sim = action_reduce(
        F.cosine_similarity(
            gc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
        ),
        action_mask,
    )
    gc_multi_action_waypts_cos_sim = action_reduce(
        F.cosine_similarity(
            torch.flatten(gc_actions[:, :, :2], start_dim=1),
            torch.flatten(batch_action_label[:, :, :2], start_dim=1),
            dim=-1,
        ),
        action_mask,
    )

    return {
        "uc_action_loss": uc_action_loss,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim,
        "gc_dist_loss": gc_dist_loss,
        "gc_action_loss": gc_action_loss,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim,
    }
