import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq
import cv2
import numpy as np
import os
import yaml

class TrajectoryProjector:
    """Handles action space conversions and camera projection.

    Loads action stats and camera intrinsics once, then provides methods to
    convert between normalized deltas, cumulative actions, and pixel coords.
    """

    base_dir = os.path.dirname(__file__)
    default_action_config = os.path.join(base_dir, "../flownav/data/data_config.yaml")
    default_camera_config = os.path.join(
        base_dir, "../thirdparty/visualnav-transformer/train/vint_train/data/data_config.yaml"
    )

    def __init__(self, dataset_name="recon", image_size=(640, 480),
                 action_config_path=default_action_config,
                 camera_config_path=default_camera_config,
                 camera_params=None):
        """
        Args:
            dataset_name: dataset key in camera config yaml.
            image_size: (width, height) for pixel clipping.
            action_config_path: path to action stats yaml.
            camera_config_path: path to camera metrics yaml.
            camera_params: dict to directly specify camera intrinsics, e.g.:
                {
                    "camera_height": 0.95,
                    "camera_x_offset": 0.45,
                    "fx": 272.5, "fy": 266.4, "cx": 320.0, "cy": 220.0,
                    "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0,
                }
                When provided, camera_config_path and dataset_name are ignored
                for camera loading.
        """
        with open(action_config_path, "r") as f:
            action_config = yaml.safe_load(f)
        self.action_stats = {k: np.array(v) for k, v in action_config["action_stats"].items()}

        if camera_params is not None:
            self._init_camera_from_dict(camera_params)
        else:
            with open(camera_config_path, "r") as f:
                camera_config = yaml.safe_load(f)
            cam = camera_config[dataset_name]["camera_metrics"]
            self._init_camera_from_yaml(cam)
        self.image_size = image_size

    def _init_camera_from_yaml(self, cam):
        """Initialize from yaml camera_metrics nested dict."""
        cm = cam["camera_matrix"]
        dc = cam["dist_coeffs"]
        self._init_camera_from_dict({
            "camera_height": cam["camera_height"],
            "camera_x_offset": cam.get("camera_x_offset", 0.0),
            "fx": cm["fx"], "fy": cm["fy"], "cx": cm["cx"], "cy": cm["cy"],
            "k1": dc["k1"], "k2": dc["k2"], "p1": dc["p1"], "p2": dc["p2"], "k3": dc["k3"],
        })

    def _init_camera_from_dict(self, p):
        self.camera_height = p["camera_height"]
        self.camera_x_offset = p.get("camera_x_offset", 0.0)
        self.camera_matrix = np.array([
            [p["fx"], 0.0, p["cx"]],
            [0.0, p["fy"], p["cy"]],
            [0.0, 0.0, 1.0],
        ])
        # 畸变矫正参数，k1,k2,k3为径向畸变，p1,p2为切向畸变
        self.dist_coeffs = np.array([
            p.get("k1", 0.0), p.get("k2", 0.0),
            p.get("p1", 0.0), p.get("p2", 0.0),
            p.get("k3", 0.0), 0.0, 0.0, 0.0,
        ])

    def ndeltas_to_actions(self, ndeltas):
        """Normalized deltas (B, T, 2) tensor → cumulative actions (B, T, 2) tensor."""
        ndeltas_np = ndeltas.detach().cpu().numpy().reshape(ndeltas.shape[0], -1, 2)
        ndeltas_np = (ndeltas_np + 1) / 2 * (self.action_stats["max"] - self.action_stats["min"]) + self.action_stats["min"]
        actions = np.cumsum(ndeltas_np, axis=1)
        return torch.from_numpy(actions).float().to(ndeltas.device)

    def project_points(self, xy):
        """Local (x, y) waypoints (B, T, 2) np → pixel (u, v) (B, T, 2) np.

        Reused from visualnav-transformer/train/vint_train/visualizing/action_utils.py.
        """
        batch_size, horizon, _ = xy.shape
        xyz = np.concatenate(
            [xy, -self.camera_height * np.ones(list(xy.shape[:-1]) + [1])], axis=-1
        )
        rvec = tvec = (0, 0, 0)
        xyz[..., 0] += self.camera_x_offset
        xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
        uv, _ = cv2.projectPoints(
            xyz_cv.reshape(batch_size * horizon, 3).astype(np.float64),
            rvec, tvec, self.camera_matrix, self.dist_coeffs,
        )
        uv = uv.reshape(batch_size, horizon, 2)
        return uv

    def actions_to_pixels(self, actions_np):
        """Cumulative actions (B, T, 2) np → clipped pixel coords (B, T, 2) np."""
        w, h = self.image_size
        uv = self.project_points(actions_np)
        uv[..., 0] = w - uv[..., 0]
        uv[..., 0] = np.clip(uv[..., 0], 0, w)
        uv[..., 1] = np.clip(uv[..., 1], 0, h)
        return uv


class FlowCorrectWrapper(nn.Module):
    """Pluggable FlowCorrect module for velocity field correction.

    Wraps a frozen NoMaD model, adding:
    - A small LoRA correction network that produces velocity adjustments

    The interface is identical to NoMaD: forward(func_name, **kwargs).
    """

    def __init__(self, base_model, encoding_dim=256, hidden_dim=64, alpha=1.0,
                 dataset_name="recon", **projector_kwargs):
        super().__init__()
        self.base_model = base_model
        self.encoding_dim = encoding_dim
        self.alpha = alpha
        self.projector = TrajectoryProjector(dataset_name=dataset_name, **projector_kwargs)

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


    def forward(self, func_name, **kwargs):
        """Drop-in replacement for NoMaD.forward()."""
        if func_name == "noise_pred_net":
            return self._corrected_velocity(**kwargs)
        return self.base_model(func_name, **kwargs)

    def _corrected_velocity(self, sample, timestep, global_cond, **_):
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

        return v_base + self.alpha * v_corr

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
            dict with:
                "ndeltas": (B, N, T, 2) normalized deltas (for flow_edit_loss).
                "actions": (B, N, T, 2) cumulative actions in local coords.
                "pixels":  (B, N, T, 2) pixel coordinates (for VLM rendering).
                "obs_cond": (B, encoding_dim) encoded observation.
        """
        if device is None:
            device = obs_images.device
        B = obs_images.shape[0]

        no_mask = torch.zeros(B, dtype=torch.long, device=device)
        obs_cond = self.base_model(
            "vision_encoder",
            obs_img=obs_images,
            goal_img=goal_images,
            input_goal_mask=no_mask,
        )
        obs_cond_rep = obs_cond.repeat_interleave(num_samples, dim=0)

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

        ndeltas = traj[-1]  # (B*N, T, 2) normalized deltas
        actions = self.projector.ndeltas_to_actions(ndeltas)  # (B*N, T, 2) cumulative
        pixels_np = self.projector.actions_to_pixels(actions.cpu().numpy())  # (B*N, T, 2)

        return {
            "ndeltas": ndeltas.reshape(B, num_samples, pred_horizon, action_dim),
            "actions": actions.reshape(B, num_samples, pred_horizon, action_dim),
            "pixels": torch.from_numpy(pixels_np).float().reshape(B, num_samples, pred_horizon, action_dim),
            "obs_cond": obs_cond,
        }

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

            # Corrected velocity (LoRA active)
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

            v_corrected = v_base + self.alpha * v_corr 

            # Weighted loss: later steps matter more
            w_n = (n + 1) / num_steps
            step_loss = w_n * F.mse_loss(v_corrected, v_target.detach())
            total_loss = total_loss + step_loss

            # Advance ODE with base model (detached, no grad through ODE path)
            with torch.no_grad():
                x = x + dt * v_base

        return total_loss / num_steps

    def flow_correct_step(
        self, obs_images, goal_images, scorer_fn,
        num_samples=5, num_steps=10, pred_horizon=8,
    ):
        """End-to-end: sample → VLM score → select best → flow_edit_loss.

        Args:
            obs_images: (B, C, H, W) observation tensor.
            goal_images: (B, C, H, W) goal tensor.
            scorer_fn: callable(obs_image_np, list_of_pixel_trajs) → list of scores.
                obs_image_np: (H, W, 3) uint8 numpy array.
                list_of_pixel_trajs: list of N arrays, each (T, 2) in pixel coords.
                Returns: list of N float scores.
            num_samples: trajectories per observation.
            num_steps: ODE integration steps.
            pred_horizon: prediction horizon.

        Returns:
            loss: scalar tensor (flow_edit_loss on best trajectories).
        """
        B = obs_images.shape[0]
        device = obs_images.device

        with torch.no_grad():
            result = self.sample_trajectories(
                obs_images, goal_images,
                pred_horizon=pred_horizon, num_samples=num_samples,
                num_steps=num_steps, use_correction=False,
            )

        obs_cond = result["obs_cond"]
        ndeltas = result["ndeltas"]     # (B, N, T, 2)
        pixels = result["pixels"]       # (B, N, T, 2)

        best_ndeltas = []
        for b in range(B):
            pixel_trajs = [pixels[b, n].cpu().numpy() for n in range(num_samples)]
            obs_np = (obs_images[b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            scores = scorer_fn(obs_np, pixel_trajs)
            best_idx = int(np.argmax(scores))
            best_ndeltas.append(ndeltas[b, best_idx])

        corrected_action = torch.stack(best_ndeltas, dim=0).to(device)  # (B, T, 2)
        return self.flow_edit_loss(obs_cond, corrected_action, num_steps=num_steps)

    def save_plugin(self, path):
        """Save only LoRA weights."""
        torch.save({
            "lora": self.lora.state_dict(),
        }, path)

    def load_plugin(self, path):
        """Load LoRA weights."""
        ckpt = torch.load(path, map_location="cpu")
        self.lora.load_state_dict(ckpt["lora"])

    def trainable_parameters(self):
        """Return only LoRA parameters (for optimizer)."""
        return list(self.lora.parameters())

    def num_trainable_params(self):
        return sum(p.numel() for p in self.trainable_parameters())

    def train_lora(self):
        """Train LoRA correction only."""
        for p in self.lora.parameters():
            p.requires_grad = True
        for p in self.base_model.parameters():
            p.requires_grad = False

if __name__ == "__main__":
    # Quick test: instantiate wrapper and check trainable params
    from flownav.models.nomad import NoMaD
    base_model = NoMaD()
    wrapper = FlowCorrectWrapper(base_model)
    print(f"Total trainable parameters in LoRA: {wrapper.num_trainable_params()}")