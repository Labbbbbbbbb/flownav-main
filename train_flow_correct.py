"""
FlowCorrect offline training script.

Loads a frozen MeanFlow base model and trains a LoRA correction network
using VLM trajectory scoring (sample → VLM score → select best → flow_edit_loss).
"""

import argparse
import os
import time

import click
import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms

from flownav.data.vint_dataset import ViNT_Dataset
from flownav.models.nomad import NoMaD, DenseNetwork
from flownav.models.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from meanflownav.models.meanflow_unet1d import MeanFlowConditionalUnet1D
from reward.flow_correct import FlowCorrectWrapper
from reward.vlm_trajectory_scorer import VLMTrajectoryScorer


def build_base_model(config):
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        depth_cfg=config["depth"],
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    noise_pred_net = MeanFlowConditionalUnet1D(
        input_dim=2,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
    )
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )
    return model


def build_dataset(config):
    datasets = []
    for dataset_name in config["datasets"]:
        ds_cfg = config["datasets"][dataset_name]
        if "test" in ds_cfg:
            dataset = ViNT_Dataset(
                data_folder=ds_cfg["data_folder"],
                data_split_folder=ds_cfg["test"],
                dataset_name=dataset_name,
                image_size=config["image_size"],
                waypoint_spacing=ds_cfg["waypoint_spacing"],
                min_dist_cat=config["distance"]["min_dist_cat"],
                max_dist_cat=config["distance"]["max_dist_cat"],
                min_action_distance=config["action"]["min_dist_cat"],
                max_action_distance=config["action"]["max_dist_cat"],
                negative_mining=True,
                len_traj_pred=config["len_traj_pred"],
                learn_angle=config["learn_angle"],
                context_size=config["context_size"],
                context_type=config["context_type"],
                end_slack=ds_cfg["end_slack"],
                goals_per_obs=ds_cfg["goals_per_obs"],
                normalize=config["normalize"],
                goal_type=config["goal_type"],
            )
            datasets.append(dataset)
    return torch.utils.data.ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fc_cfg = config["flow_correct"]

    # Build and load base model
    click.echo(click.style(">> Building base model...", fg="yellow"))
    base_model = build_base_model(config)
    ckpt_path = fc_cfg["ckpt_path"]
    click.echo(click.style(f">> Loading checkpoint: {ckpt_path}", fg="yellow"))
    state_dict = torch.load(ckpt_path, map_location="cpu")
    base_model.load_state_dict(state_dict, strict=True)
    base_model = base_model.to(device)
    base_model.eval()

    # Build FlowCorrectWrapper
    wrapper = FlowCorrectWrapper(
        base_model,
        encoding_dim=config["encoding_size"],
        hidden_dim=fc_cfg["lora_hidden_dim"],
        alpha=fc_cfg["lora_alpha"],
    )
    wrapper = wrapper.to(device)
    wrapper.train_lora()
    click.echo(click.style(f">> LoRA trainable params: {wrapper.num_trainable_params()}", fg="green"))

    optimizer = AdamW(wrapper.trainable_parameters(), lr=float(fc_cfg["lr"]))

    # Load dataset
    click.echo(click.style(">> Loading dataset...", fg="yellow"))
    dataset = build_dataset(config)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    click.echo(click.style(f">> Loaded {len(dataset)} samples", fg="cyan"))

    # VLM scorer
    vlm_scorer = VLMTrajectoryScorer(num_trajectories=fc_cfg["num_samples"])

    def scorer_fn(obs_np, pixel_trajs):
        result = vlm_scorer.score(obs_np, pixel_trajs)
        click.echo(click.style(f"   VLM scores: {result['scores']}", fg="blue"))
        return result["scores"]

    # Image transform
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Output directory
    checkpoint_dir = config["project_folder"]
    render_dir = os.path.join(checkpoint_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)

    # Training loop
    num_steps = fc_cfg["num_steps"]
    log_freq = fc_cfg["log_freq"]
    save_freq = fc_cfg["save_freq"]
    render_freq = fc_cfg["render_freq"]
    num_samples = fc_cfg["num_samples"]
    pred_horizon = config["len_traj_pred"]

    click.echo(click.style(f">> Starting training for {num_steps} steps...", fg="green", bold=True))
    step = 0
    data_iter = iter(loader)
    losses = []

    while step < num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        obs_images = batch[0].to(device)
        goal_images = batch[1].to(device)

        obs_chunks = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat([transform(c) for c in obs_chunks], dim=1)
        goal_images = transform(goal_images)

        optimizer.zero_grad()
        loss = wrapper.flow_correct_step(
            obs_images, goal_images,
            scorer_fn=scorer_fn,
            num_samples=num_samples,
            pred_horizon=pred_horizon,
        )
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        step += 1

        if step % log_freq == 0:
            avg_loss = np.mean(losses[-log_freq:])
            click.echo(click.style(
                f"   [Step {step}/{num_steps}] loss={loss_val:.4f}  avg_loss={avg_loss:.4f}",
                fg="cyan",
            ))

        if step % save_freq == 0:
            save_path = os.path.join(checkpoint_dir, f"lora_step_{step}.pt")
            wrapper.save_plugin(save_path)
            click.echo(click.style(f"   Saved LoRA checkpoint: {save_path}", fg="green"))

        if step % render_freq == 0:
            with torch.no_grad():
                result = wrapper.sample_trajectories(
                    obs_images, goal_images,
                    pred_horizon=pred_horizon,
                    num_samples=num_samples,
                )
            pixels = result["pixels"]
            pixel_trajs = [pixels[0, n].cpu().numpy() for n in range(num_samples)]
            obs_np = batch[0][0, :3].permute(1, 2, 0).numpy()
            obs_np = (obs_np * 255).clip(0, 255).astype(np.uint8)
            render_path = os.path.join(render_dir, f"step_{step}.png")
            wrapper.projector.save_render(obs_np, pixel_trajs, render_path)
            click.echo(click.style(f"   Saved render: {render_path}", fg="green"))

    # Save final checkpoint
    final_path = os.path.join(checkpoint_dir, "lora_final.pt")
    wrapper.save_plugin(final_path)
    click.echo(click.style(f">> Training complete. Final LoRA: {final_path}", fg="green", bold=True))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="FlowCorrect Offline Training")
    parser.add_argument(
        "--config", "-c",
        default="meanflownav/config/flow_correct.yaml",
        type=str,
        help="Path to flow_correct config file",
    )
    args = parser.parse_args()

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(this_file_dir, "meanflownav/config/meanflownav.yaml"), "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    click.echo(click.style(f">> Using config file: {args.config}", fg="yellow"))
    config.update(user_config)

    run_name = "flow_correct_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join("outputs", "logs", "flow_correct", run_name)
    os.makedirs(config["project_folder"])
    click.echo(click.style(f">> Project folder: {config['project_folder']}", fg="yellow"))

    main(config)
