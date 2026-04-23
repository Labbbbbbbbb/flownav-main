import os
import numpy as np
import torch
import torch.nn as nn
import yaml
import argparse
import time
import pickle
from PIL import Image as PILImage
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ROS 1 适配
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray

# FlowNav / MeanFlow 核心组件
import torchdiffeq
from flownav.training.utils import get_action
from utils import to_numpy, transform_images, load_model, msg_to_pil

# Flow_Correct /VLM Scorer 组件
from reward.flow_correct import FlowCorrectWrapper
from reward.flow_correct import TrajectoryProjector
from reward.vlm_trajectory_scorer import VLMTrajectoryScorer

# 配置路径（请根据你的实际情况检查这些路径）
TOPOMAP_IMAGES_DIR = "../../topomaps/images"
ROBOT_CONFIG_PATH = "../../config/robot.yaml"
MODEL_CONFIG_PATH = "../../config/models.yaml"

# 加载机器人基本配置
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]

# 话题名称（根据你的 Scout 机器人修改）
IMAGE_TOPIC = "/camera/color/image_raw" 
WAYPOINT_TOPIC = "/waypoint"
REACHED_GOAL_TOPIC = "/topoplan/reached_goal"

# 全局变量
context_queue = []
obs_img = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def callback_obs(msg):
    """处理来自 Realsense 的 ROS 1 图像消息"""
    global obs_img, context_queue
    # 注意：在 Python 3.10 下，我们使用 utils.py 里的 msg_to_pil 绕过 cv_bridge 兼容性问题
    obs_img = msg_to_pil(msg)
    
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)

def main(args):
    global context_size, obs_img
    
    # 1. 加载模型配置
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]

    # 2. 加载模型权重
    ckpt_path = args.ckpt
    print(f"[*] 正在从 {ckpt_path} 加载 MeanFlow 模型...")
    model = load_model(ckpt_path, model_params, device)
    model = model.to(device)
    model.eval()

    # 3. 加载拓扑地图
    topomap_dir = os.path.join(TOPOMAP_IMAGES_DIR, args.dir)
    topomap_filenames = sorted(os.listdir(topomap_dir), key=lambda x: int(x.split(".")[0]))
    topomap = []
    print(f"[*] 正在加载拓扑图: {args.dir}, 共 {len(topomap_filenames)} 节点")
    for fname in topomap_filenames:
        topomap.append(PILImage.open(os.path.join(topomap_dir, fname)))

    closest_node = 0
    goal_node = len(topomap) - 1 if args.goal_node == -1 else args.goal_node
    
    # 4. Scorer初始化
    Trajprojector = TrajectoryProjector(dataset_name="deploy")
    Scorer=VLMTrajectoryScorer()

    # 5. ROS 1 节点初始化
    rospy.init_node("MEANFLOW_NAVIGATION", anonymous=False)
    rospy.Subscriber(IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    waypoint_pub = rospy.Publisher(WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher(REACHED_GOAL_TOPIC, Bool, queue_size=1)
    
    ros_rate = rospy.Rate(RATE)
    print(f"[*] ROS 1 节点就绪。等待图像话题: {IMAGE_TOPIC}")

    # 5. 主循环
    while not rospy.is_shutdown():
        chosen_waypoint = np.zeros(4)

        if len(context_queue) > context_size:
            # 预处理观测图像
            obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
            obs_images = torch.split(obs_images, 3, dim=1)
            obs_images = torch.cat(obs_images, dim=1).to(device)
            mask = torch.zeros(1).long().to(device)

            # 局部搜索范围（Radius）
            start = max(closest_node - args.radius, 0)
            end = min(closest_node + args.radius + 1, goal_node)
            
            # 预处理目标节点图像
            goal_images = [transform_images(g_img, model_params["image_size"], center_crop=False).to(device) 
                           for g_img in topomap[start:end + 1]]
            goal_images = torch.concat(goal_images, dim=0)

            # 模型推理
            with torch.no_grad():
                # 预测距离以更新最近节点
                obs_repeat = obs_images.repeat(len(goal_images), 1, 1, 1)
                mask_repeat = mask.repeat(len(goal_images))
                obsgoal_cond = model('vision_encoder', obs_img=obs_repeat, goal_img=goal_images, input_goal_mask=mask_repeat)
                
                dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                dists = to_numpy(dists.flatten())
                min_idx = np.argmin(dists)
                closest_node = min_idx + start
                
                # 选取局部目标点
                sg_idx = min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)
                obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)
                
                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(args.num_samples, 1)
                else:
                    obs_cond = obs_cond.repeat(args.num_samples, 1, 1)

                # --- MeanFlow / FlowNav 核心：ODE 推理 ---
                noisy_action = torch.randn((args.num_samples, model_params["len_traj_pred"], 2), device=device)
                
                # 使用 Euler 方法求解 ODE (Flow Matching)
                traj = torchdiffeq.odeint(
                    lambda t, x: model.forward("noise_pred_net", sample=x, timestep=t, global_cond=obs_cond),
                    noisy_action,
                    torch.linspace(0, 1, args.k_steps, device=device),
                    atol=1e-4, rtol=1e-4, method="euler",
                )
                naction = traj[-1] # 取最终解
                naction = to_numpy(get_action(naction))

                ##使用VLM评估分数
                projected_traj = Trajprojector.project_points(naction)  #shape=(B，T，2)的uv坐标
                # 从堆叠的 context tensor 中取最后一帧，并转为 (H, W, 3) uint8
                last_obs = obs_images[0, -3:, :, :].detach().cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                last_obs = last_obs * std + mean
                last_obs = torch.clamp(last_obs, 0.0, 1.0)
                obs_img_np = (last_obs.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

                scores, _ = Scorer.score_trajectories(obs_img_np, projected_traj)  # scores shape=(B,)
                best_idx = int(np.argmax(scores))

                # 选择分数最高的轨迹对应的 waypoint 作为输出
                chosen_waypoint = naction[best_idx][args.waypoint]

            print(f"[NAV] 最近节点: {closest_node} | 距离: {dists[min_idx]:.2f} | 目标: {goal_node}")

        # 发布 Waypoint 给 pd_controller
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.tolist()
        waypoint_pub.publish(waypoint_msg)

        # 检查是否到达终点
        reached_goal = closest_node == goal_node
        goal_pub.publish(reached_goal)
        if reached_goal:
            print("[!] 到达终点！")

        ros_rate.sleep()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="meanflownav", type=str)
    parser.add_argument("--ckpt", required=True, type=str, help="模型权重路径 (.pth)")
    parser.add_argument("--dir", "-d", required=True, type=str, help="拓扑图目录名")
    parser.add_argument("--waypoint", "-w", default=2, type=int)
    parser.add_argument("--k_steps", "-k", default=10, type=int, help="ODE 求解步数")
    parser.add_argument("--radius", "-r", default=4, type=int)
    parser.add_argument("--close_threshold", "-t", default=3, type=int)
    parser.add_argument("--goal-node", "-g", default=-1, type=int)
    parser.add_argument("--num-samples", "-n", default=8, type=int)
    
    args = parser.parse_args()
    main(args)