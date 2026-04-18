import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq
import numpy as np
import os
import yaml


def _load_action_stats():
    """Load action stats without importing flownav.training.utils (avoids diffusers)."""
    cfg_path = os.path.join(
        os.path.dirname(__file__), "../flownav/data/data_config.yaml"
    )
    with open(cfg_path, "r") as f:
        data_config = yaml.safe_load(f)
    return {k: np.array(v) for k, v in data_config["action_stats"].items()}


def _get_action(ndeltas, action_stats):
    """Convert normalized deltas to cumulative actions (no diffusers dependency)."""
    ndeltas_np = ndeltas.detach().cpu().numpy().reshape(ndeltas.shape[0], -1, 2)
    # unnormalize: [-1,1] → real
    ndeltas_np = (ndeltas_np + 1) / 2 * (action_stats["max"] - action_stats["min"]) + action_stats["min"]
    actions = np.cumsum(ndeltas_np, axis=1)
    return torch.from_numpy(actions).float().to(ndeltas.device)


class FlowCorrectWrapper(nn.Module):
    """Pluggable FlowCorrect module for velocity field correction.

    Wraps a frozen NoMaD model, adding:
    - A small LoRA correction network that produces velocity adjustments
    - A gate network that decides when to apply corrections

    The interface is identical to NoMaD: forward(func_name, **kwargs).
    """

    def __init__(self, base_model, encoding_dim=256, hidden_dim=64):
        super().__init__()
        self.base_model = base_model
        self.encoding_dim = encoding_dim

        # Freeze base model
        for p in base_model.parameters():
            p.requires_grad = False

        # LoRA correction branch
        # Input: cat(sample, v_base, timestep_broadcast, global_cond_broadcast)
        # Dims:  (B, T, 2) + (B, T, 2) + (B, T, 1) + (B, T, encoding_dim)
        lora_input_dim = 2 + 2 + 1 + encoding_dim  # 261
        self.lora = nn.Sequential(
            nn.Linear(lora_input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 2),
        )
        # Zero-init last layer so initial correction is zero
        nn.init.zeros_(self.lora[-1].weight)
        nn.init.zeros_(self.lora[-1].bias)

        # Gate: global_cond → α ∈ [0, 1]
        self.gate = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, func_name, **kwargs):
        """Drop-in replacement for NoMaD.forward()."""
        if func_name == "noise_pred_net":
            return self._corrected_velocity(**kwargs)
        return self.base_model(func_name, **kwargs)

    def _corrected_velocity(self, sample, timestep, global_cond, **kwargs):
        """Compute v_base + α * v_correction."""
        B, T, _ = sample.shape

        # Base velocity (frozen, no grad)
        with torch.no_grad():
            v_base = self.base_model(
                "noise_pred_net",
                sample=sample,
                timestep=timestep,
                global_cond=global_cond,
            )

        # Gate
        alpha = self.gate(global_cond)  # (B, 1)
        alpha = alpha.unsqueeze(1)      # (B, 1, 1)

        # Build LoRA input: (sample, v_base, t, cond) along last dim
        if isinstance(timestep, (int, float)):
            t_val = torch.tensor(timestep, device=sample.device, dtype=sample.dtype)
            t_val = t_val.expand(B)
        elif timestep.dim() == 0:
            t_val = timestep.expand(B)
        else:
            t_val = timestep

        t_broadcast = t_val.view(B, 1, 1).expand(B, T, 1)
        cond_broadcast = global_cond.unsqueeze(1).expand(B, T, self.encoding_dim)
        lora_input = torch.cat([sample, v_base, t_broadcast, cond_broadcast], dim=-1)

        v_corr = self.lora(lora_input)  # (B, T, 2)

        return v_base + alpha * v_corr

    @torch.no_grad()
    def sample_trajectories(
        self,
        obs_images,
        goal_images,
        pred_horizon=8,
        action_dim=2,
        num_samples=5,
        num_steps=10,
        device=None,
        use_correction=False,
    ):
        """Sample trajectory candidates from the base (or corrected) model.

        Args:
            obs_images: (B, C, H, W) observation tensor.
            goal_images: (B, C, H, W) goal tensor.
            num_samples: trajectories per observation.
            use_correction: if True, use corrected velocity; else use base.

        Returns:
            actions: (B, num_samples, pred_horizon, action_dim) in local coords.
        """
        if device is None:
            device = obs_images.device
        B = obs_images.shape[0]

        # Encode observation + goal
        no_mask = torch.zeros(B, dtype=torch.long, device=device)
        obs_cond = self.base_model(
            "vision_encoder",
            obs_img=obs_images,
            goal_img=goal_images,
            input_goal_mask=no_mask,
        )
        obs_cond_rep = obs_cond.repeat_interleave(num_samples, dim=0)  # (B*N, enc)

        # ODE integration
        x = torch.randn(B * num_samples, pred_horizon, action_dim, device=device)
        ts = torch.linspace(0, 1, num_steps, device=device)

        forward_fn = self if use_correction else self.base_model
        traj = torchdiffeq.odeint(
            lambda t, x_t: forward_fn(
                "noise_pred_net", sample=x_t, timestep=t, global_cond=obs_cond_rep
            ),
            x,
            ts,
            method="euler",
        )
        actions = _get_action(traj[-1], _load_action_stats())  # (B*N, H, 2)
        actions = actions.reshape(B, num_samples, pred_horizon, action_dim)
        return actions

    def flow_edit_loss(self, obs_cond, corrected_action, num_steps=10):
        """Compute FlowCorrect flow-edit loss for training LoRA.

        Re-runs ODE with base model, at each step computes target velocity
        that steers toward corrected_action, and trains LoRA to match it.

        Args:
            obs_cond: (B, encoding_dim) — encoded observation (from vision_encoder).
            corrected_action: (B, pred_horizon, 2) — VLM-selected best trajectory
                              in **normalized delta** space (same as model output).
            num_steps: number of ODE integration steps.

        Returns:
            loss: scalar tensor.
        """
        B, T, D = corrected_action.shape
        device = corrected_action.device
        dt = 1.0 / num_steps

        # Start from noise
        x = torch.randn(B, T, D, device=device)
        total_loss = 0.0

        for n in range(num_steps):
            t_val = torch.tensor(n * dt, device=device)
            remaining = (num_steps - n) * dt

            # Target velocity: steer from x_n to corrected_action
            v_target = (corrected_action - x) / remaining

            # Corrected velocity (LoRA active, gate forced to 1)
            v_base = self.base_model(
                "noise_pred_net",
                sample=x.detach(),
                timestep=t_val,
                global_cond=obs_cond,
            ).detach()

            # Build LoRA input
            t_broadcast = t_val.expand(B).view(B, 1, 1).expand(B, T, 1)
            cond_broadcast = obs_cond.unsqueeze(1).expand(B, T, self.encoding_dim)
            lora_input = torch.cat(
                [x.detach(), v_base, t_broadcast, cond_broadcast], dim=-1
            )
            v_corr = self.lora(lora_input)  # (B, T, 2)

            v_corrected = v_base + v_corr  # gate=1 in stage 1

            # Weighted loss: later steps matter more
            w_n = (n + 1) / num_steps
            step_loss = w_n * F.mse_loss(v_corrected, v_target.detach())
            total_loss = total_loss + step_loss

            # Advance ODE with base model (detached, no grad through ODE path)
            with torch.no_grad():
                x = x + dt * v_base

        return total_loss / num_steps

    def grpo_loss(self, obs_cond, trajectories, scores, num_steps=10, eps=1e-6):
        """GRPO training for LoRA via advantage-weighted flow edit loss.

        Samples a group of trajectories, scores them with VLM, normalizes
        scores to advantages, and trains LoRA weighted by advantage.
        Only trajectories with positive advantage contribute (reinforce good ones).

        Args:
            obs_cond: (B, encoding_dim) — encoded observation.
            trajectories: (B, N, pred_horizon, 2) — N sampled trajectories
                          in normalized delta space.
            scores: (B, N) — VLM scores for each trajectory.
            num_steps: ODE integration steps for flow edit loss.
            eps: std epsilon for numerical stability.

        Returns:
            loss: scalar tensor.
        """
        B, N, T, D = trajectories.shape
        device = trajectories.device

        # Group-relative advantages: (score - mean) / (std + eps)
        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True)
        advantages = (scores - mean) / (std + eps)  # (B, N)

        # Only reinforce trajectories with positive advantage
        advantages = advantages.clamp(min=0)

        # Skip if no positive advantages
        if advantages.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Compute weighted flow edit loss
        total_loss = torch.tensor(0.0, device=device)
        total_weight = 0.0

        for i in range(N):
            w = advantages[:, i].mean().item()  # scalar weight for trajectory i
            if w <= 0:
                continue
            # flow_edit_loss for this trajectory across the batch
            traj_i = trajectories[:, i]  # (B, T, D)
            loss_i = self.flow_edit_loss(obs_cond, traj_i, num_steps=num_steps)
            total_loss = total_loss + w * loss_i
            total_weight += w

        if total_weight > 0:
            total_loss = total_loss / total_weight

        return total_loss

    def gate_loss(self, obs_cond, labels, entropy_weight=0.1):
        """Compute gate training loss.

        Args:
            obs_cond: (B, encoding_dim) — encoded observation.
            labels: (B,) float — 1.0 if correction needed, 0.0 if not.
            entropy_weight: λ for entropy penalty (encourages decisive 0/1).

        Returns:
            loss: scalar tensor.
        """
        alpha = self.gate(obs_cond).squeeze(-1)  # (B,)

        # BCE loss
        bce = F.binary_cross_entropy(alpha, labels)

        # Entropy penalty: encourage α close to 0 or 1
        entropy = -(alpha * torch.log(alpha + 1e-8)
                     + (1 - alpha) * torch.log(1 - alpha + 1e-8))
        entropy_penalty = entropy.mean()

        return bce - entropy_weight * entropy_penalty

    def save_plugin(self, path):
        """Save only LoRA + Gate weights (tiny file)."""
        torch.save({
            "lora": self.lora.state_dict(),
            "gate": self.gate.state_dict(),
        }, path)

    def load_plugin(self, path):
        """Load LoRA + Gate weights."""
        ckpt = torch.load(path, map_location="cpu")
        self.lora.load_state_dict(ckpt["lora"])
        self.gate.load_state_dict(ckpt["gate"])

    def trainable_parameters(self):
        """Return only LoRA + Gate parameters (for optimizer)."""
        return list(self.lora.parameters()) + list(self.gate.parameters())

    def num_trainable_params(self):
        return sum(p.numel() for p in self.trainable_parameters())

    def train_lora(self):
        """Train LoRA correction only (gate fixed α=1)."""
        for p in self.lora.parameters():
            p.requires_grad = True
        for p in self.gate.parameters():
            p.requires_grad = False

    def train_gate(self):
        """Train gate only (LoRA frozen)."""
        for p in self.lora.parameters():
            p.requires_grad = False
        for p in self.gate.parameters():
            p.requires_grad = True
