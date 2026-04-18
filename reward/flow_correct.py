import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import yaml
from PIL import Image, ImageDraw, ImageFont
from constant import TRAJ_COLORS

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

    def render(self, obs_image, pixel_trajs, best_idx=None):
        """Draw trajectories on observation image.

        Args:
            obs_image: PIL Image or numpy array (H, W, 3) uint8.
            pixel_trajs: list of N arrays, each (T, 2) in pixel coords.
            best_idx: if set, highlight this trajectory with thicker line.

        Returns:
            PIL Image with trajectories drawn.
        """
        if isinstance(obs_image, np.ndarray):
            obs_image = Image.fromarray(obs_image)
        img = obs_image.copy().convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except (IOError, OSError):
            font = ImageFont.load_default()

        for idx, traj in enumerate(pixel_trajs):
            color = TRAJ_COLORS[idx % len(TRAJ_COLORS)]
            points = [(float(x), float(y)) for x, y in traj]
            width = 5 if idx == best_idx else 2
            if len(points) >= 2:
                draw.line(points, fill=color, width=width)
            for pt in points:
                r = 4 if idx == best_idx else 2
                draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=color)
            if points:
                draw.text((points[-1][0] + 5, points[-1][1] - 7), str(idx + 1), fill=color, font=font)
        return img

    def save_render(self, obs_image, pixel_trajs, save_path, best_idx=None):
        """Render trajectories on image and save to disk.

        Args:
            obs_image: PIL Image or numpy array (H, W, 3) uint8.
            pixel_trajs: list of N arrays, each (T, 2) in pixel coords.
            save_path: output file path (e.g. "output.png").
            best_idx: if set, highlight this trajectory.
        """
        img = self.render(obs_image, pixel_trajs, best_idx=best_idx)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        img.save(save_path)
        return img


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

    def _corrected_velocity(self, sample, timestep, stoptime, global_cond, **_):
        """Compute v_base + α * v_correction."""
        B, T, _ = sample.shape

        model_unwrapped = self.base_model.module if hasattr(self.base_model, "module") else self.base_model
        with torch.no_grad():
            v_base = model_unwrapped.noise_pred_net(
                sample=sample,
                timestep=timestep,
                stoptime=stoptime,
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
        device=None,
        use_correction=False,
    ):
        """Sample trajectory candidates via one-step MeanFlow inference.

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

        e = torch.randn(B * num_samples, pred_horizon, action_dim, device=device)
        t = torch.ones(B * num_samples, device=device)
        h = torch.ones(B * num_samples, device=device)

        model_unwrapped = self.base_model.module if hasattr(self.base_model, "module") else self.base_model

        if use_correction:
            u = self._corrected_velocity(
                sample=e, timestep=t, stoptime=h, global_cond=obs_cond_rep
            )
        else:
            u = model_unwrapped.noise_pred_net(
                sample=e, timestep=t, stoptime=h, global_cond=obs_cond_rep
            )

        ndeltas = e - u  # (B*N, T, 2)
        actions = self.projector.ndeltas_to_actions(ndeltas)
        pixels_np = self.projector.actions_to_pixels(actions.cpu().numpy())

        return {
            "ndeltas": ndeltas.reshape(B, num_samples, pred_horizon, action_dim),
            "actions": actions.reshape(B, num_samples, pred_horizon, action_dim),
            "pixels": torch.from_numpy(pixels_np).float().reshape(B, num_samples, pred_horizon, action_dim),
            "obs_cond": obs_cond,
        }

    def flow_edit_loss(self, obs_cond, corrected_action):
        """Compute FlowCorrect loss for training LoRA.

        Uses random-timestep flow matching: at random t, interpolate between
        noise and corrected_action, then train corrected velocity to point
        toward corrected_action.

        Args:
            obs_cond: (B, encoding_dim) — encoded observation.
            corrected_action: (B, T, 2) — VLM-selected best trajectory
                              in normalized delta space.

        Returns:
            loss: scalar tensor.
        """
        B, T, D = corrected_action.shape
        device = corrected_action.device

        e = torch.randn(B, T, D, device=device)
        t_val = torch.rand(B, device=device)

        x_t = t_val.view(B, 1, 1) * corrected_action + (1 - t_val.view(B, 1, 1)) * e
        v_target = corrected_action - e

        stoptime = torch.ones(B, device=device)

        model_unwrapped = self.base_model.module if hasattr(self.base_model, "module") else self.base_model
        with torch.no_grad():
            v_base = model_unwrapped.noise_pred_net(
                sample=x_t.detach(),
                timestep=t_val,
                stoptime=stoptime,
                global_cond=obs_cond,
            )

        t_broadcast = t_val.view(B, 1, 1).expand(B, T, 1)
        cond_broadcast = obs_cond.unsqueeze(1).expand(B, T, self.encoding_dim)
        lora_input = torch.cat(
            [x_t.detach(), v_base, t_broadcast, cond_broadcast], dim=-1
        )
        v_corr = self.lora(lora_input)
        v_corrected = v_base + self.alpha * v_corr

        return F.mse_loss(v_corrected, v_target.detach())

    def flow_correct_step(
        self, obs_images, goal_images, scorer_fn,
        num_samples=5, pred_horizon=8,
    ):
        """End-to-end: sample → VLM score → select best → flow_edit_loss.

        Args:
            obs_images: (B, C, H, W) observation tensor.
            goal_images: (B, C, H, W) goal tensor.
            scorer_fn: callable(obs_image_np, list_of_pixel_trajs) → list of scores.
            num_samples: trajectories per observation.
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
                use_correction=False,
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
        return self.flow_edit_loss(obs_cond, corrected_action)

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
    import sys
    import yaml
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../thirdparty/consistency-policy"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../thirdparty/py-meanflow"))

    from flownav.models.nomad import NoMaD, DenseNetwork
    from flownav.models.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
    from meanflownav.models.meanflow_unet1d import MeanFlowConditionalUnet1D
    from flownav.data.vint_dataset import ViNT_Dataset
    from torch.utils.data import DataLoader
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = os.path.join(os.path.dirname(__file__), "..")

    config_path = os.path.join(project_root, "meanflownav/config/meanflownav.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(">> Building model...")
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
    base_model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )

    ckpt_path = os.path.join(project_root, "outputs/logs/meanflownav/meanflownav_2026_04_18_18_00_24/latest.pth")
    print(f">> Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    base_model.load_state_dict(state_dict, strict=True)
    base_model = base_model.to(device)
    base_model.eval()
    print(">> Model loaded successfully.")

    # Load a real data batch
    print(">> Loading dataset...")
    ds_cfg = config["datasets"]["go_stanford"]
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ViNT_Dataset(
        data_folder=ds_cfg["data_folder"],
        data_split_folder=ds_cfg["test"],
        dataset_name="go_stanford",
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
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    obs_images = batch[0].to(device)  # (B, C*context, H, W)
    goal_images = batch[1].to(device)  # (B, 3, H, W)
    print(f">> Batch loaded: obs={obs_images.shape}, goal={goal_images.shape}")

    obs_chunks = torch.split(obs_images, 3, dim=1)
    obs_images = torch.cat([transform(c) for c in obs_chunks], dim=1)
    goal_images = transform(goal_images)

    # Build FlowCorrectWrapper
    wrapper = FlowCorrectWrapper(base_model, encoding_dim=config["encoding_size"])
    wrapper = wrapper.to(device)
    print(f">> LoRA trainable params: {wrapper.num_trainable_params()}")

    # Test sample_trajectories
    print("\n>> Testing sample_trajectories...")
    num_samples = 5
    result = wrapper.sample_trajectories(
        obs_images, goal_images,
        pred_horizon=config["len_traj_pred"],
        num_samples=num_samples,
    )
    print(f"  ndeltas: {result['ndeltas'].shape}")
    print(f"  actions: {result['actions'].shape}")
    print(f"  pixels:  {result['pixels'].shape}")
    print(f"  obs_cond: {result['obs_cond'].shape}")

    # Test flow_edit_loss
    print("\n>> Testing flow_edit_loss...")
    wrapper.train_lora()
    obs_cond = result["obs_cond"]
    corrected_action = result["ndeltas"][:, 0]
    loss = wrapper.flow_edit_loss(obs_cond, corrected_action)
    print(f"  loss: {loss.item():.4f}")
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in wrapper.trainable_parameters() if p.grad is not None)
    print(f"  LoRA grad norm: {grad_norm:.4f}")

    # Test flow_correct_step
    print("\n>> Testing flow_correct_step...")
    def dummy_scorer(obs_np, pixel_trajs):
        return [float(i) for i in range(len(pixel_trajs))]

    wrapper.lora.zero_grad()
    loss = wrapper.flow_correct_step(
        obs_images, goal_images, scorer_fn=dummy_scorer,
        num_samples=num_samples, pred_horizon=config["len_traj_pred"],
    )
    print(f"  loss: {loss.item():.4f}")
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in wrapper.trainable_parameters() if p.grad is not None)
    print(f"  LoRA grad norm: {grad_norm:.4f}")

    # Test render
    print("\n>> Testing render...")
    pixels = result["pixels"]
    pixel_trajs = [pixels[0, n].cpu().numpy() for n in range(num_samples)]
    obs_np = batch[0][0, :3].permute(1, 2, 0).numpy()
    obs_np = (obs_np * 255).clip(0, 255).astype(np.uint8)
    save_path = os.path.join(project_root, "outputs/test_render.png")
    wrapper.projector.save_render(obs_np, pixel_trajs, save_path, best_idx=num_samples - 1)
    print(f"  Saved to {save_path}")

    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        wrapper.save_plugin(f.name)
        wrapper.load_plugin(f.name)
        print("\n>> save/load plugin: OK")

    print("\nAll tests passed.")