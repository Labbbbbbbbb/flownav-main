import os
import cv2

import numpy as np
import torch
import torch.nn as nn
import yaml
import argparse
import time
import pickle
import json
from datetime import datetime
from PIL import Image as PILImage,ImageDraw, ImageFont
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
# ROS 1 适配
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray,MultiArrayDimension

# FlowNav / MeanFlow 核心组件
import torchdiffeq
from flownav.training.utils import get_action
from utils import to_numpy, transform_images, load_model, remove_files_in_dir, msg_to_pil


# Flow_Correct /VLM Scorer 组件
from reward.flow_correct import TrajectoryProjector
from reward.vlm_trajectory_scorer import VLMTrajectoryScorer

# Ros Topics
from topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC)


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
OVERLAY_TOPIC = "/overlay_image"            #pub
# TRAJS_TOPIC = "/candidate_trajs"

# 全局变量
fig=None
context_queue = []
obs_img = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensor_to_rgb_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW tensor back to an RGB uint8 image for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(3, 1, 1)
    image_tensor = image_tensor[:3] * std + mean
    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
    return (image_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)



def callback_obs(msg):
    """处理来自 Realsense 的 ROS 1 图像消息"""
    # 注意：在 Python 3.10 下，我们使用 utils.py 里的 msg_to_pil 绕过 cv_bridge 兼容性问题
    obs_img = msg_to_pil(msg)
    
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)


ACTION_STATS = {
    'min': np.array([-2.5, -4]),
    'max': np.array([5, 4])
}

# --- 从你的相机参数中提取 ---
CAMERA_HEIGHT = 0.4446
CAMERA_X_OFFSET = 0.3612
CAMERA_MATRIX = np.array([
    [386.548, 0.0, 328.326],
    [0.0, 386.171, 243.573],
    [0.0, 0.0, 1.0]
])
DIST_COEFFS = np.array([-0.054461, 0.063385, 0.000409, -0.000540, -0.019851, 0.0, 0.0, 0.0])
VIZ_IMAGE_SIZE = (640, 480) # 根据 cx, cy 推断出的分辨率

COLORS = ['cyan', 'magenta', 'yellow', 'lime', 'red']
def init_visualization():
    """Initialize matplotlib figure with two subplots."""
    global fig, ax_left, ax_right
    plt.ion()  # Enable interactive mode
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('NoMaD Navigation - Real-time Visualization', fontsize=14, fontweight='bold')
    ax_left.set_title('Current Camera + Predicted Trajectories (5 samples)')
    ax_right.set_title('TopoMap Navigation Goal')
    return fig, ax_left, ax_right

def unnormalize_data(ndata, stats):
    """将模型输出从 [-1, 1] 转回 物理坐标 (米)"""
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def get_pos_pixels(points, camera_height, camera_x_offset, camera_matrix, dist_coeffs, tilt_deg=-5):
    """
    tilt_deg: 相机向下倾斜的角度（向下为负）
    """
    n_points = points.shape[0]
    xyz = np.zeros((n_points, 3))
    xyz[:, 0] = points[:, 0] + camera_x_offset 
    xyz[:, 1] = points[:, 1]                   
    xyz[:, 2] = -camera_height                 

    # 构造旋转矩阵 (绕X轴旋转，即抬头/低头)
    tilt_rad = np.radians(tilt_deg)      
    # 这里是一个简化的处理方式：调整 xyz_cv 的投影坐标
    # 更好的办法是设置 rvec = (tilt_rad, 0, 0)
    rvec = np.array([tilt_rad, 0, 0], dtype=float)
    tvec = np.array([0, 0, 0], dtype=float)
    
    # 按照训练代码的 stack 逻辑
    xyz_cv = np.stack([xyz[:, 1], -xyz[:, 2], xyz[:, 0]], axis=-1)
    
    uv, _ = cv2.projectPoints(xyz_cv, rvec, tvec, camera_matrix, dist_coeffs)
    uv = uv.reshape(n_points, 2)
    
    # 保持和训练一致的水平镜像处理
    uv[:, 0] = 640 - uv[:, 0] 
    
    return uv


def visualize_navigation(
    current_obs_img: PILImage.Image,
    topomap_img: PILImage.Image,
    predicted_trajectories: np.ndarray, # (num_samples, horizon, 2)
    closest_node: int,
    goal_node: int
):
    global fig, ax_left, ax_right
    
    if fig is None:
        init_visualization()
    
    ax_left.clear()
    ax_right.clear()
    
    # --- 左图处理 ---
    # 将 PIL 图片转为 numpy 以便获取尺寸
    obs_np = np.array(current_obs_img)
    h, w = obs_np.shape[:2]
    ax_left.imshow(current_obs_img)
    
    # 绘制每一条轨迹
    for i, traj_norm in enumerate(predicted_trajectories):
        # 1. 反归一化：把 [-1, 1] 转为米
        traj_xy_norm = traj_norm[:, :2] 

        # traj_meters = unnormalize_data(traj_norm, ACTION_STATS)
        # traj_meters = unnormalize_data(traj_xy_norm, ACTION_STATS)
        traj_meters=traj_xy_norm
        
        # 2. 投影：把 米 转为 像素坐标
        traj_pixels = get_pos_pixels(
            traj_meters, 
            CAMERA_HEIGHT, 
            CAMERA_X_OFFSET, 
            CAMERA_MATRIX, 
            DIST_COEFFS
        )
        
        # 3. 绘图
        color = COLORS[i % len(COLORS)]
        u_coords = traj_pixels[:, 0]
        v_coords = traj_pixels[:, 1]
        
        # 过滤掉图像外的点（可选）
        mask = (u_coords >= 0) & (u_coords < w) & (v_coords >= 0) & (v_coords < h)
        
        ax_left.plot(u_coords[mask], v_coords[mask], color=color, linewidth=2, 
                    marker='o', markersize=4, label=f'Traj {i+1}', alpha=0.8)
    
    ax_left.set_xlim(0, w)
    ax_left.set_ylim(h, 0) # 像素坐标系Y轴向下
    ax_left.set_title(f'Current Camera + {len(predicted_trajectories)} Predicted Trajectories')
    ax_left.legend(loc='upper right', fontsize=8)
    
    # --- 右图处理 ---
    if topomap_img is not None:
        ax_right.imshow(topomap_img)
        ax_right.set_title(f'TopoMap Goal (Node {closest_node}/{goal_node})')
    
    plt.pause(0.01)



@staticmethod
def msg_from_numpy(rgb: np.ndarray, stamp=None, frame_id="camera"):
    msg = Image()
    msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.height, msg.width = rgb.shape[:2]
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = rgb.tobytes()
    return msg

def save_images_and_actions():
    if cur_img is not None and cur_naction is not None:
        print(f"Saving Image and action {im_idx}")
        cur_img.save(f"{cur_exp_im_dir}/{im_idx}.png")
        
        with open(f"{cur_exp_pkl_dir}/{im_idx}.pkl", "wb") as f:
            pickle.dump(cur_naction, f)
            
        im_idx += 1

def main(args):
    global context_size,im_idx,cur_img,cur_naction,cur_exp_pkl_dir,cur_exp_im_dir
    
    # 1. 加载模型配置
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]
    
    exp_dir = args.exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    cur_img = None
    cur_naction = None
    ckpt_path = Path(args.ckpt)
    cur_exp_dir = f"{exp_dir}/{args.model}_{ckpt_path.name}_{args.dir}_{args.k_steps}"
    os.makedirs(cur_exp_dir, exist_ok=True)
    cur_exp_im_dir = f"{cur_exp_dir}/images"
    os.makedirs(cur_exp_im_dir, exist_ok=True)

    cur_exp_pkl_dir = f"{cur_exp_dir}/pkl"
    os.makedirs(cur_exp_pkl_dir, exist_ok=True)
    
    # 初始化轨迹消息格式
    trajs_msg = Float32MultiArray()

    ns = int(args.num_samples)
    lt = int(model_params["len_traj_pred"])
    cd = 2  # x,y

    trajs_msg.layout.dim = [
        MultiArrayDimension(label="num_samples", size=ns, stride=lt * cd),
        MultiArrayDimension(label="len_traj_pred", size=lt, stride=cd),
        MultiArrayDimension(label="coord", size=cd, stride=1),
    ]
    trajs_msg.layout.data_offset = 0

    # 2. 加载模型权重
    ckpt_path = args.ckpt
    print(f"[*] 正在从 {ckpt_path} 加载 MeanFlow 模型...")
    model = load_model(ckpt_path, model_params, device)
    model = model.to(device)
    model.eval()

    # 3. 加载拓扑地图
    dir_name = args.dir
    topomap_name_dir = os.path.join(TOPOMAP_IMAGES_DIR, dir_name)
    img_idx = 0


    if not os.path.isdir(topomap_name_dir):
        os.makedirs(topomap_name_dir)
    else:
        print(f"{topomap_name_dir} already exists. Removing previous images...")
        remove_files_in_dir(topomap_name_dir)
        
    print("Waiting for images...")


    closest_node = 0
    im_idx = 0
    
    # 4. Scorer初始化
    Trajprojector = TrajectoryProjector(dataset_name="deploy")
    Scorer=VLMTrajectoryScorer()

    
    # 5. ROS 1 节点初始化
    rospy.init_node("MEANFLOW_NAVIGATION", anonymous=False)
    rospy.Subscriber(IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    waypoint_pub = rospy.Publisher(WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)
    sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
    # trajs_pub = rospy.Publisher(TRAJS_TOPIC, Float32MultiArray, queue_size=1)
    overlay_pub = rospy.Publisher(OVERLAY_TOPIC, Image, queue_size=1)
    timer_save = rospy.Timer(rospy.Duration(1.0), lambda event: save_images_and_actions())
    ros_rate = rospy.Rate(RATE)
    print(f"[*] ROS 1 节点就绪。等待图像话题: {IMAGE_TOPIC}")
    annotated_image_msg = None

    # 8. 主循环
    while not rospy.is_shutdown():
        chosen_waypoint = np.zeros(4)

        if len(context_queue) > context_size:
            # 预处理观测图像
            obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
            obs_images = torch.split(obs_images, 3, dim=1)  #因为拼接是在通道维度上，所以这里按3在dim=1上分割
            obs_images = torch.cat(obs_images, dim=1).to(device)
            fake_goal = torch.randn((1, 3, *model_params["image_size"])).to(device)
            mask = torch.ones(1).long().to(device) # ignore the goal

            # 模型推理（带时间戳）
            with torch.no_grad():
                timeline = {}
                timeline["loop_start_ts"] = datetime.now().isoformat()
                t0 = time.perf_counter()

                # vision encoder
                t_vision_start = time.perf_counter()

                obs_cond = model('vision_encoder', obs_img=obs_images, goal_img=fake_goal, input_goal_mask=mask)
                
                t_vision_end = time.perf_counter()
                timeline["vision_start_ts"] = datetime.now().isoformat()
                timeline["vision_ms"] = (t_vision_end - t_vision_start) * 1000.0

                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(args.num_samples, 1)
                else:
                    obs_cond = obs_cond.repeat(args.num_samples, 1, 1)

                # sampling / one-step MeanFlow
                # t_sample_start = time.perf_counter()
                # noisy_action = torch.randn((args.num_samples, model_params["len_traj_pred"], 2), device=device)
                # t = torch.ones(noisy_action.shape[0], device=device)
                # h = torch.ones(noisy_action.shape[0], device=device)
                # t_noise_start = time.perf_counter()
                # u = model.noise_pred_net(sample=noisy_action, timestep=t, stoptime=h, global_cond=obs_cond)
                # t_noise_end = time.perf_counter()
                # traj = noisy_action - u
                # naction = to_numpy(get_action(traj))
                # t_sample_end = time.perf_counter()
                # timeline["noise_pred_ms"] = (t_noise_end - t_noise_start) * 1000.0
                # timeline["sampling_ms"] = (t_sample_end - t_sample_start) * 1000.0

                # sampleing / k-step MeanFlow
                k_steps = max(1, int(args.k_steps))
                t_sample_start = time.perf_counter()
                noisy_action = torch.randn((args.num_samples, model_params["len_traj_pred"], 2), device=device)
                t = torch.ones(noisy_action.shape[0], device=device)
                h = torch.ones(noisy_action.shape[0], device=device)
                t_noise_start = time.perf_counter()
                x = noisy_action
                dt = 1.0 / float(k_steps)
                for k in range(k_steps):
                    t = torch.full((x.shape[0],), 1.0 - float(k) * dt, device=device)
                    h = torch.full((x.shape[0],), dt, device=device)
                    u = model.noise_pred_net(sample=x, timestep=t, stoptime=h, global_cond=obs_cond)
                    # Treat the network output as the learned displacement over the current interval.
                    x = x - u*dt 
                t_noise_end = time.perf_counter()
                traj = x
                naction = to_numpy(get_action(traj))
                t_sample_end = time.perf_counter()
                timeline["noise_pred_ms"] = (t_noise_end - t_noise_start) * 1000.0
                timeline["sampling_ms"] = (t_sample_end - t_sample_start) * 1000.0

                # Save for logging
                cur_naction = naction
                cur_img = context_queue[-1]

                sampled_actions_msg = Float32MultiArray()
                sampled_action_message_data = np.concatenate((np.array([0]), naction.flatten()))
                sampled_actions_msg.data = sampled_action_message_data.tolist()
                sampled_actions_pub.publish(sampled_actions_msg)


                t_proj_start = time.perf_counter()
                current_img = context_queue[-1]
                visualize_navigation(
                        current_obs_img=current_img,
                        topomap_img=None,
                        predicted_trajectories=naction,
                        closest_node=-1,
                        goal_node=-1,
                    )
                t_proj_end = time.perf_counter()
                timeline["projection_ms"] = (t_proj_end - t_proj_start) * 1000.0
                
                # # projection + prepare image
                # t_proj_start = time.perf_counter()
                # projected_traj = Trajprojector.project_points(naction)
                # last_obs = obs_images[0, -3:, :, :].detach().cpu()
                # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                # last_obs = last_obs * std + mean
                # last_obs = torch.clamp(last_obs, 0.0, 1.0)
                # obs_img_np = (last_obs.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                # projected_traj = projected_traj * np.array([96.0/640.0, 96.0/480.0])
                # projected_traj[..., 0] = 96.0 - projected_traj[..., 0]
                # t_proj_end = time.perf_counter()
                # timeline["projection_ms"] = (t_proj_end - t_proj_start) * 1000.0

                # # VLM scoring (may be local or remote) -- measure end-to-end call
                # t_vlm_start = time.perf_counter()
                # score_result = Scorer.score(obs_img_np, projected_traj)
                # t_vlm_end = time.perf_counter()
                # scores = score_result.get("scores")
                # annotated_PIL = score_result.get("annotated_image")
                # timeline["vlm_call_ms"] = (t_vlm_end - t_vlm_start) * 1000.0

                # # selection
                # t_sel2_start = time.perf_counter()
                # best_idx = int(np.argmax(scores)) if scores is not None else 0
                # t_sel2_end = time.perf_counter()
                # timeline["selection_ms"] = (t_sel2_end - t_sel2_start) * 1000.0

                # t_end = time.perf_counter()
                # timeline["loop_total_ms"] = (t_end - t0) * 1000.0
                # timeline["loop_end_ts"] = datetime.now().isoformat()

                # # attach per-step timestamps if available from scorer
                # if isinstance(score_result, dict) and "timings" in score_result:
                #     timeline["scorer_timings"] = score_result["timings"]

                # print("[TIMELINE] ", json.dumps(timeline, ensure_ascii=False))
                


                # annotated_np = np.array(annotated_PIL)
                # annotated_image_msg = msg_from_numpy(annotated_np)  # 转换为 ROS 消息格式

                # if args.vis_scale != 1.0:
                #     vis_img = cv2.resize(annotated_np, None, fx=args.vis_scale, fy=args.vis_scale, interpolation=cv2.INTER_LINEAR)
                # cv2.imshow('Observation', cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                # cv2.waitKey(1)
                
                
                # 发布第一个样本的指定 waypoint
                chosen_waypoint = naction[0][args.waypoint]  #直接取第一个样本，可以加入api进行选择

        # 发布 Waypoint 给 pd_controller
        
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.tolist()
        waypoint_pub.publish(waypoint_msg)

        # 发布候选轨迹
        
        
        if annotated_image_msg is not None:
            overlay_pub.publish(annotated_image_msg) # 发布带注释的图像到 ROS 话题


        ros_rate.sleep()

    # visualizer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="meanflownav", type=str)
    parser.add_argument("--ckpt", required=True, type=str, help="模型权重路径 (.pth)")
    parser.add_argument("--dir", "-d", required=True, type=str, help="拓扑图目录名")
    parser.add_argument("--waypoint", "-w", default=2, type=int)
    parser.add_argument("--k_steps", "-k", default=3, type=int, help="ODE 求解步数")
    parser.add_argument("--radius", "-r", default=4, type=int) #原来是4
    parser.add_argument("--close_threshold", "-t", default=3, type=int)
    parser.add_argument("--goal-node", "-g", default=-1, type=int)
    parser.add_argument("--vis-scale", default=5.0, type=float, help="可视化窗口缩放倍数（1.0=原始, 1.5=放大1.5倍等）")
    parser.add_argument("--num-samples", "-n", default=8, type=int)
    parser.add_argument(
        "--exp_dir",
        "-s",
        default="explore_topomap",
        type=str,
        help="Path to store the exploration topomap",
    )

    args = parser.parse_args()
    main(args)

