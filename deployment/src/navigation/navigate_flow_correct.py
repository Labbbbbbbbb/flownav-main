"""
FlowCorrect online training node for ROS 2.

Combines navigation with online LoRA correction training:
- 15 Hz navigation timer: localize + sample trajectories (MeanFlow one-step) + publish waypoint
- Background VLM thread: score trajectories + update LoRA weights via flow_edit_loss
"""

import os
import threading
import time
import traceback

import argparse
import cv2
import numpy as np
import pickle
import torch
import yaml
from cv_bridge import CvBridge
from pathlib import Path
from PIL import Image as PILImage
from torch.optim import AdamW

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, Float32MultiArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from topic_names import (
    IMAGE_TOPIC,
    WAYPOINT_TOPIC,
    SAMPLED_ACTIONS_TOPIC,
    REACHED_GOAL_TOPIC,
)
from flownav.training.utils import get_action
from utils import to_numpy, transform_images, load_meanflow_model
from reward.flow_correct import FlowCorrectWrapper
from reward.vlm_trajectory_scorer import VLMTrajectoryScorer

TOPOMAP_IMAGES_DIR = "../topomaps/images"
ROBOT_CONFIG_PATH = "../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
CAMERA_CONFIG_PATH = "../config/camera.yaml"

with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class FlowCorrectNavigationNode(Node):
    def __init__(self, args):
        super().__init__("FlowCorrect_Navigation_Node")

        exp_dir = args.exp_dir
        os.makedirs(exp_dir, exist_ok=True)

        self.context_size = None
        self.context_queue = []
        self.cur_img = None
        self.cur_naction = None
        self.im_idx = 0

        ckpt_path = Path(args.ckpt)
        self.cur_exp_dir = f"{exp_dir}/flowcorrect_{ckpt_path.name}_{args.dir}_{args.goal_node}"
        os.makedirs(self.cur_exp_dir, exist_ok=True)
        self.cur_exp_im_dir = f"{self.cur_exp_dir}/images"
        os.makedirs(self.cur_exp_im_dir, exist_ok=True)
        self.cur_exp_pkl_dir = f"{self.cur_exp_dir}/pkl"
        os.makedirs(self.cur_exp_pkl_dir, exist_ok=True)
        self.lora_save_dir = f"{self.cur_exp_dir}/lora"
        os.makedirs(self.lora_save_dir, exist_ok=True)

        # Load model config
        with open(MODEL_CONFIG_PATH, "r") as f:
            model_paths = yaml.safe_load(f)
        model_config_path = model_paths[args.model]["config_path"]
        with open(model_config_path, "r") as f:
            model_params = yaml.safe_load(f)
        self.model_params = model_params
        self.context_size = model_params["context_size"]

        # Load base model (MeanFlow)
        if not os.path.exists(args.ckpt):
            raise FileNotFoundError(f"Model weights not found at {args.ckpt}")
        print(f"Loading MeanFlow model from {args.ckpt}")
        base_model = load_meanflow_model(args.ckpt, model_params, device)
        base_model.eval()

        # Load camera config for trajectory projection
        with open(CAMERA_CONFIG_PATH, "r") as f:
            cam_cfg = yaml.safe_load(f)
        cam = cam_cfg["deploy"]["camera_metrics"]
        camera_params = {
            "camera_height": cam["camera_height"],
            "camera_x_offset": cam["camera_x_offset"],
            "fx": cam["camera_matrix"]["fx"],
            "fy": cam["camera_matrix"]["fy"],
            "cx": cam["camera_matrix"]["cx"],
            "cy": cam["camera_matrix"]["cy"],
            "k1": cam["dist_coeffs"]["k1"],
            "k2": cam["dist_coeffs"]["k2"],
            "p1": cam["dist_coeffs"]["p1"],
            "p2": cam["dist_coeffs"]["p2"],
            "k3": cam["dist_coeffs"]["k3"],
        }

        # Build FlowCorrectWrapper
        self.wrapper = FlowCorrectWrapper(
            base_model,
            encoding_dim=model_params["encoding_size"],
            camera_params=camera_params,
        )
        self.wrapper = self.wrapper.to(device)
        self.wrapper.train_lora()

        if args.lora_ckpt and os.path.exists(args.lora_ckpt):
            self.wrapper.load_plugin(args.lora_ckpt)
            print(f"Loaded LoRA from {args.lora_ckpt}")

        print(f"LoRA trainable params: {self.wrapper.num_trainable_params()}")

        # Optimizer
        self.optimizer = AdamW(
            self.wrapper.trainable_parameters(), lr=float(args.lora_lr)
        )

        # VLM scorer
        self.vlm_scorer = VLMTrajectoryScorer(num_trajectories=args.num_samples)
        self.num_samples = args.num_samples
        self.lora_save_freq = args.lora_save_freq

        # Shared state for async VLM scoring
        self._vlm_lock = threading.Lock()
        self._vlm_input = None
        self._vlm_best_idx = 0
        self._vlm_step = 0
        self._running = True

        # Start VLM background thread
        self._vlm_thread = threading.Thread(target=self._vlm_scoring_loop, daemon=True)
        self._vlm_thread.start()

        # Load topomap
        topomap_filenames = sorted(
            os.listdir(os.path.join(TOPOMAP_IMAGES_DIR, args.dir)),
            key=lambda x: int(x.split(".")[0]),
        )
        topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"
        num_nodes = len(os.listdir(topomap_dir))
        topomap = []
        for i in range(num_nodes):
            image_path = os.path.join(topomap_dir, topomap_filenames[i])
            topomap.append(PILImage.open(image_path))

        assert -1 <= args.goal_node < len(topomap), "Invalid goal index"
        if args.goal_node == -1:
            goal_node = len(topomap) - 1
        else:
            goal_node = args.goal_node

        self.closest_node = 0
        self.goal_node = goal_node
        self.topomap = topomap
        self.reached_goal = False
        self.br = CvBridge()
        self.args = args

        # ROS 2 subscriptions and publishers
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.image_sub = self.create_subscription(
            CompressedImage, IMAGE_TOPIC, self.callback_obs, qos_profile=qos
        )
        self.waypoint_pub = self.create_publisher(
            Float32MultiArray, WAYPOINT_TOPIC, qos_profile=qos
        )
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, SAMPLED_ACTIONS_TOPIC, qos_profile=qos
        )
        self.goal_pub = self.create_publisher(Bool, REACHED_GOAL_TOPIC, 1)
        self.timer = self.create_timer(
            1.0 / RATE, lambda: self.run_navigation_loop(args)
        )
        self.imsave_timer = self.create_timer(
            1, lambda: self.save_images_and_actions()
        )

        print("FlowCorrect navigation node ready. Waiting for image observations...")

    def callback_obs(self, msg):
        self.obs_img = self.br.compressed_imgmsg_to_cv2(msg)
        self.obs_img = PILImage.fromarray(
            cv2.cvtColor(self.obs_img, cv2.COLOR_BGR2RGB)
        )
        if self.context_size is not None:
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(self.obs_img)
            else:
                self.context_queue.pop(0)
                self.context_queue.append(self.obs_img)

    def save_images_and_actions(self):
        if self.cur_img is not None and self.cur_naction is not None:
            self.cur_img.save(f"{self.cur_exp_im_dir}/{self.im_idx}.png")
            with open(f"{self.cur_exp_pkl_dir}/{self.im_idx}.pkl", "wb") as f:
                pickle.dump(self.cur_naction, f)
            self.im_idx += 1

    def run_navigation_loop(self, args):
        chosen_waypoint = np.zeros(4)

        if len(self.context_queue) > self.context_size:
            obs_images = transform_images(
                self.context_queue,
                self.model_params["image_size"],
                center_crop=False,
            )
            obs_images = torch.split(obs_images, 3, dim=1)
            obs_images = torch.cat(obs_images, dim=1)
            obs_images = obs_images.to(device)
            mask = torch.zeros(1).long().to(device)

            # Localization (same as navigate.py)
            start = max(self.closest_node - args.radius, 0)
            end = min(self.closest_node + args.radius + 1, self.goal_node)
            goal_image = [
                transform_images(
                    g_img, self.model_params["image_size"], center_crop=False
                ).to(device)
                for g_img in self.topomap[start : end + 1]
            ]
            goal_image = torch.concat(goal_image, dim=0)

            obsgoal_cond = self.wrapper.base_model(
                "vision_encoder",
                obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
                goal_img=goal_image,
                input_goal_mask=mask.repeat(len(goal_image)),
            )
            dists = self.wrapper.base_model(
                "dist_pred_net", obsgoal_cond=obsgoal_cond
            )
            dists = to_numpy(dists.flatten())
            min_idx = np.argmin(dists)
            self.closest_node = min_idx + start

            # Select goal conditioning
            selected_idx = min(
                min_idx + int(dists[min_idx] < args.close_threshold),
                len(goal_image) - 1,
            )
            selected_goal = goal_image[selected_idx : selected_idx + 1]

            # Sample trajectories via one-step MeanFlow
            with torch.no_grad():
                result = self.wrapper.sample_trajectories(
                    obs_images,
                    selected_goal,
                    pred_horizon=self.model_params["len_traj_pred"],
                    num_samples=self.num_samples,
                    use_correction=True,
                )

            ndeltas = result["ndeltas"]  # (1, N, T, 2)
            actions = result["actions"]  # (1, N, T, 2)
            pixels = result["pixels"]  # (1, N, T, 2)

            # Use best trajectory from last VLM scoring round
            with self._vlm_lock:
                best_idx = min(self._vlm_best_idx, self.num_samples - 1)

            naction = to_numpy(actions[0])  # (N, T, 2)

            self.cur_naction = naction
            self.cur_img = self.context_queue[-1]

            # Publish all sampled actions
            sampled_actions_msg = Float32MultiArray()
            message_data = np.concatenate(
                (np.array([best_idx]), naction.flatten())
            )
            sampled_actions_msg.data = message_data.tolist()
            self.sampled_actions_pub.publish(sampled_actions_msg)

            chosen_waypoint = naction[best_idx][args.waypoint]

            # Post data to VLM scoring thread
            obs_pil = self.context_queue[-1].resize(
                tuple(self.model_params["image_size"])
            )
            obs_np = np.array(obs_pil)
            with self._vlm_lock:
                self._vlm_input = {
                    "obs_cond": result["obs_cond"],
                    "ndeltas": ndeltas,
                    "pixels": pixels,
                    "obs_np": obs_np,
                }

        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.flatten().tolist()
        self.waypoint_pub.publish(waypoint_msg)

        reached_goal = self.closest_node == self.goal_node
        goal_reached_msg = Bool()
        goal_reached_msg.data = bool(reached_goal)
        self.goal_pub.publish(goal_reached_msg)

        if reached_goal:
            print("Reached goal! Stopping...")

    def _vlm_scoring_loop(self):
        """Background thread: VLM scoring + LoRA update."""
        while self._running:
            with self._vlm_lock:
                data = self._vlm_input
                self._vlm_input = None

            if data is None:
                time.sleep(0.1)
                continue

            try:
                obs_cond = data["obs_cond"]
                ndeltas = data["ndeltas"]  # (1, N, T, 2)
                pixels = data["pixels"]
                obs_np = data["obs_np"]

                pixel_trajs = [
                    pixels[0, n].cpu().numpy() for n in range(self.num_samples)
                ]

                result = self.vlm_scorer.score(obs_np, pixel_trajs)
                scores = result["scores"]
                best_idx = int(np.argmax(scores))

                with self._vlm_lock:
                    self._vlm_best_idx = best_idx

                print(
                    f"[VLM] scores={scores}, best={best_idx}"
                )

                # LoRA update
                corrected_action = ndeltas[0, best_idx].unsqueeze(0).to(device)
                self.optimizer.zero_grad()
                loss = self.wrapper.flow_edit_loss(obs_cond, corrected_action)
                loss.backward()
                self.optimizer.step()

                self._vlm_step += 1
                print(f"[VLM] step={self._vlm_step}, loss={loss.item():.4f}")

                if self._vlm_step % self.lora_save_freq == 0:
                    save_path = os.path.join(
                        self.lora_save_dir, f"lora_step_{self._vlm_step}.pt"
                    )
                    self.wrapper.save_plugin(save_path)
                    print(f"[VLM] Saved LoRA: {save_path}")

            except Exception as e:
                print(f"[VLM] Error: {e}")
                traceback.print_exc()
                time.sleep(1.0)

    def destroy_node(self):
        self._running = False
        if self._vlm_thread.is_alive():
            self._vlm_thread.join(timeout=5.0)
        # Save final LoRA
        final_path = os.path.join(self.lora_save_dir, "lora_final.pt")
        self.wrapper.save_plugin(final_path)
        print(f"Saved final LoRA: {final_path}")
        super().destroy_node()


def main(args):
    rclpy.init()
    node = FlowCorrectNavigationNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FlowCorrect online training + navigation"
    )
    parser.add_argument(
        "--model", "-m", default="meanflownav", type=str,
        help="Model config key in models.yaml",
    )
    parser.add_argument(
        "--ckpt", required=True, type=str,
        help="Path to MeanFlow base model checkpoint",
    )
    parser.add_argument(
        "--lora-ckpt", default=None, type=str,
        help="Path to pre-trained LoRA checkpoint (optional)",
    )
    parser.add_argument(
        "--lora-lr", default=0.001, type=float,
        help="LoRA learning rate",
    )
    parser.add_argument(
        "--lora-save-freq", default=50, type=int,
        help="Save LoRA every N VLM scoring rounds",
    )
    parser.add_argument(
        "--num-samples", "-n", default=5, type=int,
        help="Number of trajectory candidates per step",
    )
    parser.add_argument(
        "--waypoint", "-w", default=2, type=int,
        help="Index of waypoint used for navigation",
    )
    parser.add_argument(
        "--dir", required=True, type=str,
        help="Topomap directory name",
    )
    parser.add_argument(
        "--goal-node", "-g", default=-1, type=int,
        help="Goal node index in topomap (-1 for last)",
    )
    parser.add_argument(
        "--close-threshold", "-t", default=3, type=int,
        help="Distance threshold for advancing to next node",
    )
    parser.add_argument(
        "--radius", "-r", default=4, type=int,
        help="Localization radius in topomap",
    )
    parser.add_argument(
        "--exp_dir", default="./nav_experiments", type=str,
        help="Path to log experiment results",
    )
    args = parser.parse_args()
    print(f"Using {device}")
    main(args)
