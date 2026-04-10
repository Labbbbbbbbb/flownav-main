"""
MeanFlow evaluation — uses one-step inference instead of CFM + ODE.

Key changes from flownav/training/evaluate.py:
- Removes ConditionalFlowMatcher, uses one-step MeanFlow inference for eval losses
- ema_model is a plain nn.Module (not diffusers EMAModel)
- compute_losses uses MeanFlow single-step model_output
"""

import itertools

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import tqdm
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms

from flownav.data.data_utils import VISUALIZATION_IMAGE_SIZE
from flownav.training.logger import Logger
from flownav.training.utils import (
    ACTION_STATS,
    get_delta,
    normalize_data,
    from_numpy,
    visualize_action_distribution,
)
from meanflownav.training.utils import compute_losses


def evaluate(
    eval_type: str,
    ema_model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
):
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    ema_model.eval()

    num_batches = len(dataloader)

    # Loggers
    uc_action_loss_logger = Logger(
        "uc_action_loss", eval_type, window_size=print_log_freq
    )
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", eval_type, window_size=print_log_freq)
    gc_action_loss_logger = Logger(
        "gc_action_loss", eval_type, window_size=print_log_freq
    )
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    num_batches = max(int(num_batches * eval_fraction), 1)

    with tqdm.tqdm(
        itertools.islice(dataloader, num_batches),
        total=num_batches,
        dynamic_ncols=True,
        desc=f"Evaluating {eval_type} for epoch {epoch}",
        leave=True,
        colour="blue",
    ) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image,
                goal_image,
                actions,
                distance,
                goal_pos,
                _,
                action_mask,
            ) = data

            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(
                obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1]
            )
            batch_viz_goal_images = TF.resize(
                goal_image, VISUALIZATION_IMAGE_SIZE[::-1]
            )
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)
            action_mask = action_mask.to(device)
            distance = distance.to(device)

            if i % print_log_freq == 0 and print_log_freq != 0:
                losses = compute_losses(
                    ema_model=ema_model,
                    batch_obs_images=batch_obs_images,
                    batch_goal_images=batch_goal_images,
                    batch_dist_label=distance.to(device),
                    batch_action_label=actions.to(device),
                    device=device,
                    action_mask=action_mask.to(device),
                    use_wandb=use_wandb,
                )

                for key, value in losses.items():
                    if key in loggers:
                        loggers[key].log_data(value.item())

                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    print(
                        f"(epoch {epoch}) (batch {i}/{num_batches - 1}) "
                        f"{logger.display()}"
                    )

                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_action_distribution(
                    ema_model=ema_model,
                    batch_obs_images=batch_obs_images,
                    batch_goal_images=batch_goal_images,
                    batch_viz_obs_images=batch_viz_obs_images,
                    batch_viz_goal_images=batch_viz_goal_images,
                    batch_action_label=actions,
                    batch_distance_labels=distance,
                    batch_goal_pos=goal_pos,
                    device=device,
                    eval_type=eval_type,
                    project_folder=project_folder,
                    epoch=epoch,
                    num_images_log=num_images_log,
                    num_samples=4,
                    use_wandb=use_wandb,
                )
