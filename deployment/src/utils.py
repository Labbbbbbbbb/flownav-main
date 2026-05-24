import os
import sys
import io
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image as PILImage
from typing import List, Tuple, Dict, Optional
import yaml
import shutil

# ROS 1 消息类型
from sensor_msgs.msg import Image

# 核心模型导入 - 注意这里指向你安装的 flownav 包
from flownav.models.nomad import NoMaD, DenseNetwork
from flownav.models.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from meanflownav.models.meanflow_unet1d import MeanFlowConditionalUnet1D
from flownav.data.data_utils import IMAGE_ASPECT_RATIO

def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """针对 MeanFlow/FlowNav 优化的模型加载函数"""
    
    # 1. 实例化视觉编码器 (包含深度配置)
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        depth_cfg=config["depth"] # 关键：必须传入深度图配置
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    
    # 2. 实例化动作预测网络 (Meanflownav-Net)
    noise_pred_net = MeanFlowConditionalUnet1D(
        input_dim=2,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
        use_v_head=True,   # ← 加这个
    )
    
    # 3. 实例化距离预测网络
    dist_pred_net = DenseNetwork(embedding_dim=config["encoding_size"])
        
    # 4. 组装模型
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_net,
    )

    # 5. 加载权重
    print(f"[*] 正在加载权重文件: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    return model

def msg_to_pil(msg: Image) -> PILImage.Image:
    """适配 ROS 1 的原生图像转换，不依赖 cv_bridge"""
    try:
        # 处理常用的 bgr8 (Realsense 默认) 或 rgb8
        if msg.encoding == 'rgb8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        elif msg.encoding == 'bgr8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            img = img[:, :, ::-1] # BGR 转 RGB
        else:
            # 兜底方案：尝试通用转换
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            
        return PILImage.fromarray(img)
    except Exception as e:
        print(f"[!] 图像转换失败: {e}")
        return None

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def transform_images(pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
    """图像预处理：Resize + Normalize"""
    transform_type = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if not isinstance(pil_imgs, list):
        pil_imgs = [pil_imgs]
        
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        
        pil_img = pil_img.resize(image_size) 
        transf_img = transform_type(pil_img)
        transf_imgs.append(torch.unsqueeze(transf_img, 0))
        
    return torch.cat(transf_imgs, dim=1)

def clip_angle(theta) -> float:
    """角度归一化到 [-pi, pi]"""
    theta %= 2 * np.pi
    if -np.pi < theta < np.pi:
        return theta
    return theta - 2 * np.pi

def remove_files_in_dir(dir_path: str):
    """清理目录"""
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")