import os
import sys
import numpy as np
import torch
import yaml
import pickle
import argparse
import torchdiffeq
from pathlib import Path
from PIL import Image as PILImage

# ROS 1
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32MultiArray
from cv_bridge import CvBridge

# reuse existing utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../deployment/src"))
from utils import to_numpy, transform_images, load_model, clip_angle

from flownav.training.utils import get_action

# ---------- constants ----------
TOPOMAP_IMAGES_DIR = "/media/zhanyt/Data/Zytisworking/flownav-main/DataSet/nomad_dataset/custom/bags_traj_002_0"
ROBOT_CONFIG_PATH = "../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"

with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]
DT = 1.0 / RATE
EPS = 1e-8

IMAGE_TOPIC = "/camera/image_raw"
CMD_VEL_TOPIC = "/cmd_vel"
SAMPLED_ACTIONS_TOPIC = "/sampled_actions"
REACHED_GOAL_TOPIC = "/topoplan/reached_goal"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- PD controller ----------
def pd_controller(waypoint: np.ndarray):
    assert len(waypoint) in (2, 4)
    if len(waypoint) == 2:
        dx, dy = waypoint
    else:
        dx, dy, hx, hy = waypoint
    if len(waypoint) == 4 and abs(dx) < EPS and abs(dy) < EPS:
        v = 0.0
        w = clip_angle(np.arctan2(hy, hx)) / DT
    elif abs(dx) < EPS:
        v = 0.0
        w = np.sign(dy) * np.pi / (2 * DT)
    else:
        v = dx / DT
        w = np.arctan(dy / dx) / DT
    v = float(np.clip(v, 0, MAX_V))
    w = float(np.clip(w, -MAX_W, MAX_W))
    return v, w


# ---------- main node ----------
class ScoutNavigationNode:
    def __init__(self, args: argparse.Namespace):
        rospy.init_node("scout_navigation_node", anonymous=False)

        self.args = args
        self.br = CvBridge()
        self.context_queue = []
        self.cur_img = None
        self.cur_naction = None
        self.im_idx = 0
        self.reached_goal = False

        # experiment dirs
        ckpt_path = Path(args.ckpt)
        self.cur_exp_dir = os.path.join(
            args.exp_dir,
            f"{args.model}_{ckpt_path.name}_{args.dir}_{args.goal_node}_{args.k_steps}",
        )
        self.cur_exp_im_dir = os.path.join(self.cur_exp_dir, "images")
        self.cur_exp_pkl_dir = os.path.join(self.cur_exp_dir, "pkl")
        for d in (self.cur_exp_dir, self.cur_exp_im_dir, self.cur_exp_pkl_dir):
            os.makedirs(d, exist_ok=True)

        # load model
        with open(MODEL_CONFIG_PATH, "r") as f:
            model_paths = yaml.safe_load(f)
        model_config_path = model_paths[args.model]["config_path"]
        with open(model_config_path, "r") as f:
            self.model_params = yaml.safe_load(f)
        self.context_size = self.model_params["context_size"]

        if not os.path.exists(args.ckpt):
            raise FileNotFoundError(f"Model weights not found at {args.ckpt}")
        print(f"Loading model from {args.ckpt}")
        self.model = load_model(args.ckpt, self.model_params, device)
        self.model.eval()

        # load topomap
        topomap_dir = os.path.join(TOPOMAP_IMAGES_DIR, args.dir)
        filenames = sorted(os.listdir(topomap_dir), key=lambda x: int(x.split(".")[0]))
        self.topomap = [PILImage.open(os.path.join(topomap_dir, f)) for f in filenames]
        self.closest_node = 0
        assert -1 <= args.goal_node < len(self.topomap), "Invalid goal index"
        self.goal_node = len(self.topomap) - 1 if args.goal_node == -1 else args.goal_node

        # ROS pub/sub
        self.cmd_vel_pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=1)
        self.sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
        self.goal_pub = rospy.Publisher(REACHED_GOAL_TOPIC, Bool, queue_size=1)
        rospy.Subscriber(IMAGE_TOPIC, Image, self.callback_obs, queue_size=1)

        rospy.Timer(rospy.Duration(1.0 / RATE), self.run_navigation_loop)
        rospy.Timer(rospy.Duration(1.0), self.save_images_and_actions)

        print("Waiting for image observations...")

    def callback_obs(self, msg: Image):
        img_cv = self.br.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        pil_img = PILImage.fromarray(img_cv)
        if len(self.context_queue) < self.context_size + 1:
            self.context_queue.append(pil_img)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(pil_img)

    def save_images_and_actions(self, _event=None):
        if self.cur_img is not None and self.cur_naction is not None:
            print(f"Saving image and action {self.im_idx}")
            self.cur_img.save(f"{self.cur_exp_im_dir}/{self.im_idx}.png")
            with open(f"{self.cur_exp_pkl_dir}/{self.im_idx}.pkl", "wb") as f:
                pickle.dump(self.cur_naction, f)
            self.im_idx += 1

    def run_navigation_loop(self, _event=None):
        args = self.args
        twist = Twist()

        if len(self.context_queue) <= self.context_size:
            self.cmd_vel_pub.publish(twist)
            return

        obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=False)
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1).to(device)
        mask = torch.zeros(1).long().to(device)

        start = max(self.closest_node - args.radius, 0)
        end = min(self.closest_node + args.radius + 1, self.goal_node)
        goal_image = [
            transform_images(g, self.model_params["image_size"], center_crop=False).to(device)
            for g in self.topomap[start:end + 1]
        ]
        goal_image = torch.cat(goal_image, dim=0)

        n = len(goal_image)
        obsgoal_cond = self.model(
            "vision_encoder",
            obs_img=obs_images.repeat(n, 1, 1, 1),
            goal_img=goal_image,
            input_goal_mask=mask.repeat(n),
        )
        dists = to_numpy(self.model("dist_pred_net", obsgoal_cond=obsgoal_cond).flatten())
        min_idx = int(np.argmin(dists))
        self.closest_node = min_idx + start

        with torch.no_grad():
            obs_cond = obsgoal_cond[
                min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)
            ].unsqueeze(0)
            if obs_cond.dim() == 2:
                obs_cond = obs_cond.repeat(args.num_samples, 1)
            else:
                obs_cond = obs_cond.repeat(args.num_samples, 1, 1)

            noisy_action = torch.randn(
                (args.num_samples, self.model_params["len_traj_pred"], 2), device=device
            )
            traj = torchdiffeq.odeint(
                lambda t, x: self.model.forward("noise_pred_net", sample=x, timestep=t, global_cond=obs_cond),
                noisy_action,
                torch.linspace(0, 1, args.k_steps, device=device),
                atol=1e-4, rtol=1e-4, method="euler",
            )
            naction = to_numpy(get_action(traj[-1]))

        self.cur_naction = naction
        self.cur_img = self.context_queue[-1]

        # publish sampled actions
        msg = Float32MultiArray()
        msg.data = np.concatenate(([0], naction.flatten())).tolist()
        self.sampled_actions_pub.publish(msg)

        # PD control → cmd_vel
        chosen_waypoint = naction[0][args.waypoint]
        v, w = pd_controller(chosen_waypoint)
        print(f"CHOSEN WAYPOINT: {chosen_waypoint}  v={v:.3f}  w={w:.3f}")

        reached_goal = self.closest_node == self.goal_node
        self.goal_pub.publish(Bool(data=bool(reached_goal)))

        if reached_goal:
            print("Reached goal! Stopping...")
            self.cmd_vel_pub.publish(Twist())
            rospy.signal_shutdown("goal reached")
            return

        twist.linear.x = v
        twist.angular.z = w
        self.cmd_vel_pub.publish(twist)

    def spin(self):
        rospy.spin()


def main():
    parser = argparse.ArgumentParser(description="FlowNav navigation on Scout (ROS 1)")
    parser.add_argument("--model", "-m", default="flownav", type=str)
    parser.add_argument("--ckpt", required=True, type=str, help="Checkpoint path")
    parser.add_argument("--waypoint", "-w", default=2, type=int)
    parser.add_argument("--k_steps", "-k", default=10, type=int)
    parser.add_argument("--dir", "-d", required=True, type=str, help="Topomap subdir name")
    parser.add_argument("--goal-node", "-g", default=-1, type=int)
    parser.add_argument("--close-threshold", "-t", default=3, type=int)
    parser.add_argument("--radius", "-r", default=4, type=int)
    parser.add_argument("--num-samples", "-n", default=8, type=int)
    parser.add_argument("--exp_dir", default="./nav_experiments", type=str)
    args = parser.parse_args()

    node = ScoutNavigationNode(args)
    node.spin()


if __name__ == "__main__":
    main()
