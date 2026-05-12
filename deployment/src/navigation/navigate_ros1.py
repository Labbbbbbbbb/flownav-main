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

# ROS 1 适配
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray,MultiArrayDimension

# FlowNav / MeanFlow 核心组件
import torchdiffeq
from flownav.training.utils import get_action
from utils import to_numpy, transform_images, load_model, msg_to_pil

# Flow_Correct /VLM Scorer 组件
from reward.flow_correct import TrajectoryProjector
from reward.vlm_trajectory_scorer import VLMTrajectoryScorer

# Visualization
# from visualize_goals import GoalVisualizer


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
OVERLAY_TOPIC = "/overlay_image"            #pub
# TRAJS_TOPIC = "/candidate_trajs"

# 全局变量
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
    global obs_img, context_queue
    # 注意：在 Python 3.10 下，我们使用 utils.py 里的 msg_to_pil 绕过 cv_bridge 兼容性问题
    obs_img = msg_to_pil(msg)
    
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)
            
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

def main(args):
    global context_size, obs_img, trajs_msg
    
    # 1. 加载模型配置
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]
    
    
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
    
    # 5. 可视化初始化
    # visualizer = GoalVisualizer(display=True, save=False)
    
    # 6. ROS 1 节点初始化
    rospy.init_node("MEANFLOW_NAVIGATION", anonymous=False)
    rospy.Subscriber(IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    waypoint_pub = rospy.Publisher(WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher(REACHED_GOAL_TOPIC, Bool, queue_size=1)
    # trajs_pub = rospy.Publisher(TRAJS_TOPIC, Float32MultiArray, queue_size=1)
    overlay_pub = rospy.Publisher(OVERLAY_TOPIC, Image, queue_size=1)
    
    ros_rate = rospy.Rate(RATE)
    print(f"[*] ROS 1 节点就绪。等待图像话题: {IMAGE_TOPIC}")
    annotated_image_msg = None

    # 7. 主循环
    while not rospy.is_shutdown():
        chosen_waypoint = np.zeros(4)

        if len(context_queue) > context_size:
            # 预处理观测图像
            obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
            obs_images = torch.split(obs_images, 3, dim=1)  #因为拼接是在通道维度上，所以这里按3在dim=1上分割
            obs_images = torch.cat(obs_images, dim=1).to(device)
            mask = torch.zeros(1).long().to(device)

            # 局部搜索范围（Radius）（要求topomap的图片是有序的）
            start = max(closest_node - args.radius, 0)
            end = min(closest_node + args.radius + 1, goal_node)
            
            # 预处理目标节点图像 即在当前位置上可以作为下一个目标的图片
            goal_images = [transform_images(g_img, model_params["image_size"], center_crop=False).to(device) 
                           for g_img in topomap[start:end + 1]]
            goal_images = torch.concat(goal_images, dim=0)

            # 模型推理（带时间戳）
            with torch.no_grad():
                timeline = {}
                timeline["loop_start_ts"] = datetime.now().isoformat()
                t0 = time.perf_counter()

                # vision encoder
                t_vision_start = time.perf_counter()
                obs_repeat = obs_images.repeat(len(goal_images), 1, 1, 1)
                mask_repeat = mask.repeat(len(goal_images))
                obsgoal_cond = model('vision_encoder', obs_img=obs_repeat, goal_img=goal_images, input_goal_mask=mask_repeat)
                t_vision_end = time.perf_counter()
                timeline["vision_start_ts"] = datetime.now().isoformat()
                timeline["vision_ms"] = (t_vision_end - t_vision_start) * 1000.0

                # distance prediction
                t_dist_start = time.perf_counter()
                dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                t_dist_end = time.perf_counter()
                dists = to_numpy(dists.flatten())
                timeline["dist_start_ts"] = datetime.now().isoformat()
                timeline["dist_ms"] = (t_dist_end - t_dist_start) * 1000.0

                min_idx = np.argmin(dists)
                closest_node = min_idx + start

                # choose subgoal and prepare obs_cond
                t_select_start = time.perf_counter()
                sg_idx = min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)
                obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)
                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(args.num_samples, 1)
                else:
                    obs_cond = obs_cond.repeat(args.num_samples, 1, 1)
                t_select_end = time.perf_counter()
                timeline["select_ms"] = (t_select_end - t_select_start) * 1000.0

                # sampling / one-step MeanFlow
                t_sample_start = time.perf_counter()
                noisy_action = torch.randn((args.num_samples, model_params["len_traj_pred"], 2), device=device)
                t = torch.ones(noisy_action.shape[0], device=device)
                h = torch.ones(noisy_action.shape[0], device=device)
                t_noise_start = time.perf_counter()
                u = model.noise_pred_net(sample=noisy_action, timestep=t, stoptime=h, global_cond=obs_cond)
                t_noise_end = time.perf_counter()
                traj = noisy_action - u
                naction = to_numpy(get_action(traj))
                t_sample_end = time.perf_counter()
                timeline["noise_pred_ms"] = (t_noise_end - t_noise_start) * 1000.0
                timeline["sampling_ms"] = (t_sample_end - t_sample_start) * 1000.0

                # projection + prepare image
                t_proj_start = time.perf_counter()
                projected_traj = Trajprojector.project_points(naction)
                last_obs = obs_images[0, -3:, :, :].detach().cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                last_obs = last_obs * std + mean
                last_obs = torch.clamp(last_obs, 0.0, 1.0)
                obs_img_np = (last_obs.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                projected_traj = projected_traj * np.array([96.0/640.0, 96.0/480.0])
                projected_traj[..., 0] = 96.0 - projected_traj[..., 0]
                t_proj_end = time.perf_counter()
                timeline["projection_ms"] = (t_proj_end - t_proj_start) * 1000.0

                # VLM scoring (may be local or remote) -- measure end-to-end call
                t_vlm_start = time.perf_counter()
                score_result = Scorer.score(obs_img_np, projected_traj)
                t_vlm_end = time.perf_counter()
                scores = score_result.get("scores")
                annotated_PIL = score_result.get("annotated_image")
                timeline["vlm_call_ms"] = (t_vlm_end - t_vlm_start) * 1000.0

                # selection
                t_sel2_start = time.perf_counter()
                best_idx = int(np.argmax(scores)) if scores is not None else 0
                t_sel2_end = time.perf_counter()
                timeline["selection_ms"] = (t_sel2_end - t_sel2_start) * 1000.0

                t_end = time.perf_counter()
                timeline["loop_total_ms"] = (t_end - t0) * 1000.0
                timeline["loop_end_ts"] = datetime.now().isoformat()

                # attach per-step timestamps if available from scorer
                if isinstance(score_result, dict) and "timings" in score_result:
                    timeline["scorer_timings"] = score_result["timings"]

                print("[TIMELINE] ", json.dumps(timeline, ensure_ascii=False))
                
                
                # draw=ImageDraw.Draw(annotated_PIL)
                # font = ImageFont.load_default()
                # draw.text(
                #     (5, 5),
                #     str(best_idx),
                #     fill="magenta",
                #     font=font,
                # )
                
                annotated_np = np.array(annotated_PIL)
                annotated_image_msg = msg_from_numpy(annotated_np)  # 转换为 ROS 消息格式


                # 在主循环内，在 annotated_image_msg = msg_from_numpy(annotated_np) 之后加：
                goal_img_np = tensor_to_rgb_uint8(goal_images[sg_idx])
                if annotated_np.shape[0] != goal_img_np.shape[0]:
                    goal_img_np = cv2.resize(goal_img_np, (goal_img_np.shape[1], annotated_np.shape[0]))

                vis_img = np.hstack([
                    annotated_np,
                    goal_img_np,
                ])
                if args.vis_scale != 1.0:
                    vis_img = cv2.resize(vis_img, None, fx=args.vis_scale, fy=args.vis_scale, interpolation=cv2.INTER_LINEAR)
                cv2.imshow('Observation (left) vs Goal (right)', cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
                
                # 实时可视化：观测图 + 目标图
                # visualizer.update(annotated_np, goal_images[sg_idx])

                # 选择分数最高的轨迹对应的 waypoint 作为输出
                # chosen_waypoint = naction[best_idx][args.waypoint]
                
                # 发布第一个样本的指定 waypoint
                chosen_waypoint = naction[0][args.waypoint]  #直接取第一个样本，可以加入api进行选择

            print(f"[NAV] 最近节点: {closest_node} | 距离: {dists[min_idx]:.2f} | 目标: {goal_node}")

        # 发布 Waypoint 给 pd_controller
        
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.tolist()
        waypoint_pub.publish(waypoint_msg)

        # 发布候选轨迹
        
        # trajs_msg.data = naction.astype(np.float32).reshape(-1).tolist()
        # trajs_pub.publish(trajs_msg)
        

        
        if annotated_image_msg is not None:
            overlay_pub.publish(annotated_image_msg) # 发布带注释的图像到 ROS 话题
        
        # 检查是否到达终点
        reached_goal = closest_node == goal_node
        goal_pub.publish(reached_goal)
        if reached_goal:
            print("[!] 到达终点！")

        ros_rate.sleep()

    # visualizer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="meanflownav", type=str)
    parser.add_argument("--ckpt", required=True, type=str, help="模型权重路径 (.pth)")
    parser.add_argument("--dir", "-d", required=True, type=str, help="拓扑图目录名")
    parser.add_argument("--waypoint", "-w", default=2, type=int)
    parser.add_argument("--k_steps", "-k", default=10, type=int, help="ODE 求解步数")
    parser.add_argument("--radius", "-r", default=4, type=int) #原来是4
    parser.add_argument("--close_threshold", "-t", default=3, type=int)
    parser.add_argument("--goal-node", "-g", default=-1, type=int)
    parser.add_argument("--vis-scale", default=5.0, type=float, help="可视化窗口缩放倍数（1.0=原始, 1.5=放大1.5倍等）")
    parser.add_argument("--num-samples", "-n", default=8, type=int)
    
    args = parser.parse_args()
    main(args)

    # try:
    #     main(args)
    # except KeyboardInterrupt:
    #     print("[!] 中断输入，退出...")
    # except Exception as e:
    #     print(f"[!] 错误: {e}")
    #     import traceback
    #     traceback.print_exc()