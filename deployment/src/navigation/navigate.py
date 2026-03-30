
import os
import numpy as np
import torch
import yaml
from cv_bridge import CvBridge          # ROS 图像消息 ↔ OpenCV 格式转换
import cv2
import pickle                           # 序列化保存动作数据
from PIL import Image as PILImage
import argparse
import torchdiffeq                      # ODE 积分器，用于 Flow Matching 推理
from pathlib import Path

# ROS 2 Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, Float32MultiArray
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy, QoSHistoryPolicy

# ROS Topics
from topic_names import (IMAGE_TOPIC,           # 相机图像话题名
                        WAYPOINT_TOPIC,          # 发布目标路径点的话题名
                        SAMPLED_ACTIONS_TOPIC,   # 发布所有采样轨迹的话题名
                        REACHED_GOAL_TOPIC)      # 发布是否到达目标的话题名

# Custom Imports
from flownav.training.utils import get_action    # delta → 绝对坐标轨迹
from utils import to_numpy, transform_images, load_model


# CONSTANTS
# TOPOMAP_IMAGES_DIR = "../topomaps/images"
# ROBOT_CONFIG_PATH ="../config/robot.yaml"
# MODEL_CONFIG_PATH = "../config/models.yaml"
TOPOMAP_IMAGES_DIR = "../topomaps/topomaps"  #zyt
ROBOT_CONFIG_PATH ="../../config/robot.yaml"
MODEL_CONFIG_PATH = "../../config/models.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)             # 加载机器人配置（最大速度、帧率等）
MAX_V = robot_config["max_v"]                    # 最大线速度
MAX_W = robot_config["max_w"]                    # 最大角速度
RATE = robot_config["frame_rate"]                # 控制循环频率（Hz）


# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class NavigationNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__('Navigation_Node')

        exp_dir = args.exp_dir
        os.makedirs(exp_dir, exist_ok=True)      # 创建实验根目录

        self.context_size = None                 # 历史帧数，从模型配置读取
        self.context_queue = []                  # 滑动窗口，存储最近 context_size+1 帧图像

        self.cur_img = None                      # 当前帧图像（用于保存日志）
        self.cur_naction = None                  # 当前预测的归一化动作（用于保存日志）

        self.k_steps = args.k_steps              # ODE 积分步数

        ckpt_path = Path(args.ckpt)
        #self.cur_exp_dir = f"{exp_dir}/{args.model}_{ckpt_path.name}_{args.dir}_{args.goal_node}_{args.k_steps}"
        self.cur_exp_dir = f"{exp_dir}"          # 本次实验的保存目录（zyt 简化版）
        os.makedirs(self.cur_exp_dir, exist_ok=True)

        self.cur_exp_im_dir = f"{self.cur_exp_dir}/images"
        os.makedirs(self.cur_exp_im_dir, exist_ok=True)   # 图像保存子目录

        self.cur_exp_pkl_dir = f"{self.cur_exp_dir}/pkl"
        os.makedirs(self.cur_exp_pkl_dir, exist_ok=True)  # 动作 pkl 保存子目录

        self.im_idx = 0                          # 保存文件的递增索引

        # 加载模型配置（从 models.yaml 找到对应模型的 config 路径）
        with open(MODEL_CONFIG_PATH, "r") as f:
            model_paths = yaml.safe_load(f)

        model_config_path = model_paths[args.model]["config_path"]
        with open(model_config_path, "r") as f:
            model_params = yaml.safe_load(f)     # 加载模型超参数（context_size、image_size 等）

        self.context_size = model_params["context_size"]  # 历史帧数

        # 加载模型权重
        ckpth_path = args.ckpt
        if os.path.exists(ckpth_path):
            print(f"Loading model from {ckpth_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
        self.model = load_model(
            ckpth_path,
            model_params,
            device,
        )
        self.model = self.model.to(device)
        self.model.eval()                        # 推理模式，关闭 Dropout/BN 训练行为

        # 加载拓扑地图（topomap）：一系列按顺序排列的节点图像
        # topomap_filenames = sorted(os.listdir(os.path.join(
        #     TOPOMAP_IMAGES_DIR, args.dir)), key=lambda x: int(x.split(".")[0]))
        topomap_filenames = sorted(os.listdir(
            TOPOMAP_IMAGES_DIR), key=lambda x: int(x.split(".")[0]))  # 按文件名数字排序
        # topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{args.dir}"
        topomap_dir = f"{TOPOMAP_IMAGES_DIR}"    # zyt 简化版，直接用根目录
        num_nodes = len(os.listdir(topomap_dir)) # 拓扑地图节点总数
        topomap = []
        for i in range(num_nodes):
            image_path = os.path.join(topomap_dir, topomap_filenames[i])
            topomap.append(PILImage.open(image_path))  # 将每个节点图像加载到内存

        closest_node = 0                         # 初始化：当前最近节点为起点
        assert -1 <= args.goal_node < len(topomap), "Invalid goal index"
        if args.goal_node == -1:
            goal_node = len(topomap) - 1         # -1 表示以最后一个节点为目标
        else:
            goal_node = args.goal_node
        self.reached_goal = False

        # ROS 2 订阅/发布设置
        self.image_sub = self.create_subscription(
            #CompressedImage, IMAGE_TOPIC, self.callback_obs, ...
            Image, IMAGE_TOPIC, self.callback_obs, qos_profile = QoSProfile(  # 订阅相机图像（zyt 改为非压缩格式）
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10))
        self.waypoint_pub = self.create_publisher(
            Float32MultiArray, WAYPOINT_TOPIC, qos_profile = QoSProfile(      # 发布选定路径点（x, y, sin, cos）
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10))
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, SAMPLED_ACTIONS_TOPIC, qos_profile = QoSProfile(  # 发布所有采样轨迹（用于可视化）
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10))
        self.goal_pub = self.create_publisher(Bool, REACHED_GOAL_TOPIC, 1)    # 发布是否到达目标

        # 主导航定时器：以 RATE Hz 频率调用导航循环
        self.timer = self.create_timer(1.0 / RATE, lambda: self.run_navigation_loop(args))

        # 图像/动作保存定时器：每秒保存一次当前帧和动作
        self.imsave_timer = self.create_timer(1, lambda:self.save_images_and_actions())

        print("Waiting for image observations...")

        self.model_params = model_params
        self.closest_node = closest_node
        self.goal_node = goal_node
        self.topomap = topomap
        self.br = CvBridge()                     # 初始化 ROS↔OpenCV 图像转换器

    def callback_obs(self, msg: Image):
        """ROS 图像回调：将收到的图像转换格式并维护历史帧队列"""
        self.get_logger().info("Reached Image callback!")
        #self.obs_img = self.br.compressed_imgmsg_to_cv2(msg)
        self.obs_img = self.br.imgmsg_to_cv2(msg)                              # ROS Image → OpenCV BGR
        self.obs_img = PILImage.fromarray(cv2.cvtColor(self.obs_img, cv2.COLOR_BGR2RGB))  # BGR → RGB PIL Image

        if self.context_size is not None:
            if len(self.context_queue) < self.context_size + 1:
                self.context_queue.append(self.obs_img)    # 队列未满，直接追加
            else:
                self.context_queue.pop(0)                  # 队列已满，弹出最旧帧
                self.context_queue.append(self.obs_img)    # 追加最新帧，保持滑动窗口

    def save_images_and_actions(self):
        """定时保存当前帧图像和预测动作到磁盘（用于离线分析）"""
        if self.cur_img is not None and self.cur_naction is not None:
            print(f"Saving Image and action {self.im_idx}")
            self.cur_img.save(f"{self.cur_exp_im_dir}/{self.im_idx}.png")      # 保存 PNG 图像

            with open(f"{self.cur_exp_pkl_dir}/{self.im_idx}.pkl", "wb") as f:
                pickle.dump(self.cur_naction, f)                               # 序列化保存动作数组

            self.im_idx += 1

    def run_navigation_loop(self, args):
        """主导航循环：定位当前节点 → 推理轨迹 → 发布路径点"""
        chosen_waypoint = np.zeros(4)            # 默认路径点为零（队列未满时发布停止指令）

        if len(self.context_queue) > self.context_size:  # 历史帧足够才开始推理

            # ── 图像预处理 ────────────────────────────────────────────────────
            obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=False)
            obs_images = torch.split(obs_images, 3, dim=1)   # 按 3 通道拆分为单帧列表
            obs_images = torch.cat(obs_images, dim=1)        # 重新拼接（保持通道顺序）
            obs_images = obs_images.to(device)
            mask = torch.zeros(1).long().to(device)          # goal_mask=0：有条件（保留目标）

            # ── 拓扑定位：在当前节点附近 radius 范围内找距离最近的节点 ────────
            start = max(self.closest_node - args.radius, 0)
            end = min(self.closest_node + args.radius + 1, self.goal_node)
            # 将候选范围内的所有节点图像预处理并拼成 batch
            goal_image = [transform_images(g_img, self.model_params["image_size"], center_crop=False).to(device) for g_img in self.topomap[start:end + 1]]
            goal_image = torch.concat(goal_image, dim=0)

            # 视觉编码：将当前观测与每个候选节点配对，得到条件特征
            obsgoal_cond = self.model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
            # 距离预测：预测当前观测到每个候选节点的时间距离
            dists = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dists = to_numpy(dists.flatten())
            min_idx = np.argmin(dists)               # 找距离最小的节点索引（相对于 start）
            self.closest_node = min_idx + start      # 更新全局最近节点

            # ── 轨迹推理 ─────────────────────────────────────────────────────
            with torch.no_grad():
                # 若当前节点距下一节点足够近（< close_threshold），则超前一步选目标节点
                obs_cond = obsgoal_cond[min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)].unsqueeze(0)

                # 复制条件特征以支持多样本并行采样
                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(args.num_samples, 1)
                else:
                    obs_cond = obs_cond.repeat(args.num_samples, 1, 1)

                # 从标准正态分布采样初始噪声轨迹
                noisy_action = torch.randn(
                    (args.num_samples, self.model_params["len_traj_pred"], 2), device=device)

                # ODE 积分：从噪声出发，沿速度场积分 k_steps 步，得到去噪轨迹
                traj = torchdiffeq.odeint(
                    lambda t, x: self.model.forward("noise_pred_net", sample=x, timestep=t, global_cond=obs_cond),
                    noisy_action,                                    # 初始状态（纯噪声）
                    torch.linspace(0, 1, self.k_steps, device=device),  # 时间步序列 [0→1]
                    atol=1e-4,
                    rtol=1e-4,
                    method="euler",                                  # 一阶欧拉法
                )
                naction = traj[-1]                                   # 取最后一步（t=1），traj[-1]是归一化后的位移增量

                # 反归一化 + cumsum 还原为绝对坐标轨迹
                naction = to_numpy(get_action(naction))          # 从增量恢复出相对位置坐标，形状为 [num_samples, len_traj_pred, 2]  （相对当前位置的位置偏移量

                # 保存当前帧和动作用于日志
                self.cur_naction = naction
                self.cur_img = self.context_queue[-1]

                # 发布所有采样轨迹（第一个元素为标志位 0，后接展平的轨迹数据）
                sampled_actions_msg = Float32MultiArray()
                message_data = np.concatenate((np.array([0]), naction.flatten()))
                sampled_actions_msg.data = message_data.tolist()
                print("published sampled actions")
                self.sampled_actions_pub.publish(sampled_actions_msg)

                naction = naction[0]                                 # 取第一个样本的轨迹
                chosen_waypoint = naction[args.waypoint]             # 取第 waypoint 个路径点作为控制目标

        # 发布选定路径点（无论是否推理，都发布，未推理时为零向量）
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.flatten().tolist()
        self.waypoint_pub.publish(waypoint_msg)

        print(f"CHOSEN WAYPOINT: {chosen_waypoint}")

        # 判断是否到达目标节点并发布
        reached_goal = self.closest_node == self.goal_node
        goal_reached_msg = Bool()
        goal_reached_msg.data = bool(reached_goal)
        self.goal_pub.publish(goal_reached_msg)

        if reached_goal:
            print("Reached goal! Stopping...")


def main(args: argparse.Namespace):
    rclpy.init()                                 # 初始化 ROS 2 运行时
    navigation_node = NavigationNode(args)

    try:
        rclpy.spin(navigation_node)              # 进入事件循环，持续处理回调
    except KeyboardInterrupt:
        pass
    finally:
        navigation_node.destroy_node()           # 销毁节点，释放资源
        rclpy.shutdown()                         # 关闭 ROS 2 运行时


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run FlowNav Navigation on the turtlebot")

    parser.add_argument(
        "--model",
        "-m",
        default="flownav",
        type=str,
        help="Model to run: Only FlowNav is supported currently (default: flownav)",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        type=str,
        help="Checkpoint path",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2,
        type=int,
        help="index of the waypoint used for navigation (default: 2)",  # 从预测的 T 步轨迹中取第几步作为控制目标
    )
    parser.add_argument(
        "--k_steps",
        "-k",
        default=10,
        type=int,
        help="Number of time steps",             # ODE 积分步数，越多越精确但越慢
    )
    parser.add_argument(
        "--dir",
        "-d",
        required=True,
        type=str,
        help="path to topomap images",           # 拓扑地图图像目录
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="goal node index in the topomap (default: -1)",  # -1 表示最后一个节点
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="temporal distance within the next node in the topomap before localizing to it",
        # 当预测距离 < 此阈值时，认为已足够接近当前节点，超前选下一个节点为目标
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="temporal number of locobal nodes to look at in the topopmap for localization",
        # 定位时在当前节点前后各搜索 radius 个节点
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help="Number of actions sampled from the exploration model",  # 并行采样的轨迹数量
    )
    parser.add_argument(
        "--exp_dir",
        #"-d",   #  #by zyt, -d is used for topomap dir
        default="./nav_experiments",
        type=str,
        help="Path to log experiment results",   # 实验日志保存根目录
    )

    args = parser.parse_args()

    print(f"Using {device}")
    main(args)
