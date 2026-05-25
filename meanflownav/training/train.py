"""
MeanFlow training step — replaces CFM with JVP-based loss.

Key differences from flownav/training/train.py:
- Dual timestep (t, r) sampling instead of ConditionalFlowMatcher
- JVP loss: V = u - (t-r) * du/dt.detach(), loss = (V - (e-naction))^2
- Adaptive weighting with norm_p and norm_eps
- MeanFlow EMA (periodic update) instead of diffusers EMAModel
- noise_pred_net called directly with (timestep=t, stoptime=h)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import tqdm
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from meanflownav.models.time_sampler import sample_two_timesteps
from meanflownav.models.ema import update_ema_net

from flownav.data.data_utils import VISUALIZATION_IMAGE_SIZE
from flownav.training.logger import Logger
from flownav.training.utils import (
    ACTION_STATS,
    get_delta,
    normalize_data,
    from_numpy,
)
from meanflownav.training.utils import compute_losses
from meanflownav.training.utils import visualize_action_distribution
from meanflownav.training.utils import visualize_flow_stage_distribution

def train(
    model: nn.Module,
    ema_model: nn.Module,
    num_updates: torch.Tensor,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    meanflow_args,
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    num_batches = len(dataloader)

    # Unwrap DataParallel if needed
    model_unwrapped = model.module if hasattr(model, "module") else model

    # Loggers
    flow_loss_logger = Logger("flow_loss", "train", window_size=print_log_freq)
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    loggers = {
        "flow_loss": flow_loss_logger,
        "dist_loss": dist_loss_logger,
        "total_loss": total_loss_logger,
    }

    with tqdm.tqdm(
        dataloader,
        desc=f"Train epoch {epoch}",
        leave=True,
        dynamic_ncols=True,
        colour="magenta",
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

            # Split the observation image into RGB channels
            obs_images = torch.split(obs_image, 3, dim=1)
            batch_viz_obs_images = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1])
            batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)

            # Get action mask
            action_mask = action_mask.to(device)

            # Get naction and normalize it
            deltas = get_delta(actions)
            print(f"[debug] deltas shape: {deltas.shape}, min: {deltas.min().item():.4f}, max: {deltas.max().item():.4f}, mean: {deltas.mean().item():.4f}, std: {deltas.std().item():.4f}")
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)

            # Get batch size
            B = actions.shape[0]

            # Generate random goal mask
            goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            obsgoal_cond = model(
                "vision_encoder",
                obs_img=batch_obs_images,
                goal_img=batch_goal_images,
                input_goal_mask=goal_mask,
            )

            # Get distance label
            distance = distance.float().to(device)

            # Predict distance
            dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)
            dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / (
                1e-2 + (1 - goal_mask.float()).mean()
            )

            # ============================================================
            # MeanFlow JVP loss (replaces CFM lines 134-145 in original)
            # ============================================================
            e = torch.randn(naction.shape, device=device)
            t, r = sample_two_timesteps(
                meanflow_args, num_samples=B, device=device
            )
            # Broadcast for (B, 8, 2) — 3D, not 4D like images
            t = t.view(-1, 1, 1)
            r = r.view(-1, 1, 1)

            z = (1 - t) * naction + t * e

            # Direct reference to noise_pred_net for JVP
            noise_pred_net = model_unwrapped.noise_pred_net

            def u_func(z, t, r):
                h = t - r
                return noise_pred_net(
                    sample=z,
                    timestep=t.view(-1),
                    stoptime=h.view(-1),
                    global_cond=obsgoal_cond,
                )
            v_pred = noise_pred_net.forward_v(z, t.view(-1), global_cond=obsgoal_cond)
            with torch.amp.autocast("cuda", enabled=False):
                aux_loss = (v_pred - (e - naction)) ** 2
                aux_loss = aux_loss.sum(dim=(1, 2))
                aux_adp_wt = (aux_loss.detach() + meanflow_args.norm_eps) ** meanflow_args.norm_p
                aux_flow_loss = (aux_loss / aux_adp_wt * action_mask).mean() / (action_mask.mean() + 1e-2)

            v = v_pred.detach()
            
            dtdt = torch.ones_like(t)
            drdt = torch.zeros_like(r)

            with torch.amp.autocast("cuda", enabled=False):
                u_pred, dudt = torch.func.jvp(
                    u_func, (z, t, r), (v, dtdt, drdt)
                )

                V=u_pred - (t - r) * (dudt.detach())
                loss = (V-(e - naction)) ** 2
                loss = loss.sum(dim=(1, 2))  # (B,) — squared L2

                # Adaptive weighting
                adp_wt = (
                    loss.detach() + meanflow_args.norm_eps
                ) ** meanflow_args.norm_p
                flow_loss = loss / adp_wt

                # Apply action mask
                flow_loss = (flow_loss * action_mask).mean() / (
                    action_mask.mean() + 1e-2
                )

            # Total loss
            total_loss = alpha * dist_loss + (1 - alpha) * (flow_loss + aux_flow_loss)  #0.1*aux_flow_loss

            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update MeanFlow EMA
            num_updates += 1
            update_ema_net(model_unwrapped, ema_model, num_updates.item())

            # Logging
            loss_cpu = total_loss.item()
            tepoch.set_postfix(loss=loss_cpu)

            if use_wandb:
                wandb.log(
                    {
                        "total_loss": loss_cpu,
                        "dist_loss": dist_loss.item(),
                        "flow_loss": flow_loss.item(),
                    }
                )

            for logger in loggers.values():
                if logger.name == "flow_loss":
                    logger.log_data(flow_loss.item())
                elif logger.name == "dist_loss":
                    logger.log_data(dist_loss.item())
                elif logger.name == "total_loss":
                    logger.log_data(loss_cpu)

            if i % print_log_freq == 0 and print_log_freq != 0:
                for key, logger in loggers.items():
                    print(
                        f"(epoch {epoch}) (batch {i}/{num_batches - 1}) "
                        f"{logger.display()}"
                    )

            if image_log_freq != 0 and i % image_log_freq == 0:
                if print_log_freq != 0 and i % print_log_freq == 0:
                    actions_xy = actions[..., :2].float()
                    goal_pos_xy = goal_pos[..., :2].float() if goal_pos.shape[-1] >= 2 else goal_pos.float()

                    print(
                        f"[viz-debug] epoch={epoch} batch={i} "
                        f"ema_model.training={ema_model.training} "
                        f"model.training={model.training}"
                    )
                    print(
                        f"[viz-debug] actions_xy min/max=({actions_xy.min().item():.4f}, {actions_xy.max().item():.4f}) "
                        f"mean/std=({actions_xy.mean().item():.4f}, {actions_xy.std().item():.4f})"
                    )
                    print(
                        f"[viz-debug] goal_pos min/max=({goal_pos_xy.min().item():.4f}, {goal_pos_xy.max().item():.4f}) "
                        f"mean/std=({goal_pos_xy.mean().item():.4f}, {goal_pos_xy.std().item():.4f})"
                    )
                    print(
                        f"[viz-debug] distance min/max=({distance.min().item():.4f}, {distance.max().item():.4f}) "
                        f"mean/std=({distance.mean().item():.4f}, {distance.std().item():.4f})"
                    )
                # visualize_action_distribution(
                #     ema_model=ema_model,
                #     batch_obs_images=batch_obs_images,
                #     batch_goal_images=batch_goal_images,
                #     batch_viz_obs_images=batch_viz_obs_images,
                #     batch_viz_goal_images=batch_viz_goal_images,
                #     batch_action_label=actions,
                #     batch_distance_labels=distance,
                #     batch_goal_pos=goal_pos,
                #     device=device,
                #     eval_type="train",
                #     project_folder=project_folder,
                #     epoch=epoch,
                #     num_images_log=num_images_log,
                #     num_samples=4,
                #     use_wandb=use_wandb,
                # )
                # visualize_flow_stage_distribution(
                #     ema_model=ema_model,
                #     batch_obs_images=batch_obs_images,
                #     batch_goal_images=batch_goal_images,
                #     batch_action_label=actions,
                #     device=device,
                #     eval_type="train",
                #     project_folder=project_folder,
                #     epoch=epoch,
                #     num_images_log=num_images_log,
                #     num_samples=4,
                #     use_wandb=use_wandb,
                # )       




                