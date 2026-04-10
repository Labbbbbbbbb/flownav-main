"""
MeanFlow training loop — uses MeanFlow EMA instead of diffusers EMAModel.

Key changes from flownav/training/loop.py:
- EMA: init_ema + update_ema_net (periodic, double precision)
- num_updates buffer for EMA tracking
- Saves ema_model state_dict directly
"""

import copy
import os
from typing import Dict

import click
import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from meanflownav.models.ema import init_ema
from meanflownav.training.evaluate import evaluate
from meanflownav.training.train import train


def main_loop(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    meanflow_args,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
) -> None:
    latest_path = os.path.join(project_folder, "latest.pth")

    # Create MeanFlow EMA model
    model_unwrapped = model.module if hasattr(model, "module") else model
    ema_model = copy.deepcopy(model_unwrapped)
    ema_model = init_ema(
        model_unwrapped, ema_model, meanflow_args.ema_decay
    )
    ema_model.to(device)

    # EMA update counter
    num_updates = torch.tensor(0, dtype=torch.long, device=device)

    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            click.echo(
                click.style(
                    f"> Start epoch {epoch}/{current_epoch + epochs - 1}",
                    fg="magenta",
                )
            )
            train(
                model=model,
                ema_model=ema_model,
                num_updates=num_updates,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                goal_mask_prob=goal_mask_prob,
                project_folder=project_folder,
                epoch=epoch,
                meanflow_args=meanflow_args,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                alpha=alpha,
            )
            lr_scheduler.step()

        # Save checkpoints
        numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
        torch.save(ema_model.state_dict(), numbered_path)
        torch.save(ema_model.state_dict(), os.path.join(project_folder, "ema_latest.pth"))

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(model_unwrapped.state_dict(), numbered_path)
        torch.save(model_unwrapped.state_dict(), latest_path)

        torch.save(optimizer.state_dict(), os.path.join(project_folder, "optimizer_latest.pth"))
        torch.save(lr_scheduler.state_dict(), os.path.join(project_folder, "scheduler_latest.pth"))

        # Evaluate
        if (epoch + 1) % eval_freq == 0:
            for dataset_type in test_dataloaders:
                click.echo(
                    click.style(
                        f"> Evaluating {dataset_type} at epoch {epoch}",
                        fg="blue",
                    )
                )
                evaluate(
                    eval_type=dataset_type,
                    ema_model=ema_model,
                    dataloader=test_dataloaders[dataset_type],
                    transform=transform,
                    device=device,
                    goal_mask_prob=goal_mask_prob,
                    project_folder=project_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    num_images_log=num_images_log,
                    wandb_log_freq=wandb_log_freq,
                    use_wandb=use_wandb,
                    eval_fraction=eval_fraction,
                )

        if use_wandb:
            wandb.log({"lr": optimizer.param_groups[0]["lr"]}, commit=False)

    if use_wandb:
        wandb.log({})
