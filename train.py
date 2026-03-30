'''
根目录 train.py — 训练启动器（入口脚本）

解析命令行参数、加载 YAML 配置
初始化 GPU、随机种子、数据集、DataLoader
搭建完整模型（视觉编码器 + UNet + 距离头）
加载预训练权重（断点续训 / Depth-Anything）
配置优化器、学习率调度器、wandb
最后调用 main_loop() 把控制权交出去
它是**"组装一切然后开跑"**的入口，本身不含任何训练逻辑。

flownav/training/train.py — 单 epoch 训练执行器

只有一个函数 train()，被 main_loop 在每个 epoch 调用
实现了真正的训练循环：
图像预处理（split obs、resize、normalize）
动作归一化（get_delta → normalize_data）
随机生成 goal mask（以概率 goal_mask_prob 遮蔽目标，实现无条件/有条件混合训练）
前向传播：视觉编码 → 距离预测 → Flow Matching（用 ConditionalFlowMatcher 采样 t, xt, ut，预测速度场 vt）
计算损失：loss = α * dist_loss + (1-α) * flow_loss
反向传播 + EMA 权重更新
按频率打印日志、上传 wandb、可视化动作分布
核心区别一句话：

根目录 train.py 是配置和启动，flownav/training/train.py 是每个 batch 实际发生的事。前者调用后者（通过 main_loop 间接调用）。
此外mainloop里面还调用了flownav/training/evaluate.py中的evaluate()函数，在每隔eval_freq个epoch进行一次评估，评估时会调用evaluate()函数对测试集进行评估并记录结果。

'''



import argparse          # 解析命令行参数
import os                # 文件/目录操作
import time              # 获取当前时间（用于命名实验文件夹）

import click             # 彩色终端输出
import numpy as np       # 随机种子设置
import torch             # PyTorch 核心
import torch.backends.cudnn as cudnn  # cuDNN 加速配置
import torch.nn as nn    # 神经网络模块（DataParallel 多卡训练）
import wandb             # 实验追踪与可视化平台
import yaml              # 读取 YAML 配置文件
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D  # 扩散策略中的条件 UNet，用于预测轨迹噪声
from torch.optim import AdamW                          # AdamW 优化器（带权重衰减）
from torch.utils.data import ConcatDataset, DataLoader # 合并多个数据集 / 批量加载数据
from torchvision import transforms                     # 图像预处理变换
from flownav.data.vint_dataset import ViNT_Dataset     # FlowNav 自定义数据集类
from flownav.models.nomad import DenseNetwork, NoMaD   # 完整模型 NoMaD 和距离预测头 DenseNetwork
from flownav.models.nomad_vint import NoMaD_ViNT, replace_bn_with_gn  # 视觉编码器 + 将 BatchNorm 替换为 GroupNorm（多卡训练更稳定）
from flownav.training.loop import main_loop            # 训练主循环
from warmup_scheduler import GradualWarmupScheduler    # 学习率预热调度器


def main(config: dict) -> None:
    # ── 1. 设备配置 ──────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 按物理总线顺序编号 GPU，保证 ID 与 nvidia-smi 一致
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]                      # 默认使用第 0 块 GPU
        elif isinstance(config["gpu_ids"], int):
            config["gpu_ids"] = [config["gpu_ids"]]      # 单个整数转为列表，统一格式
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]          # 将 GPU ID 列表转为逗号分隔字符串，告知 CUDA 只暴露这些 GPU
        )
        click.echo(
            click.style(f">> Using GPUs: {config['gpu_ids']}", fg="green", bold=True)
        )
    else:
        click.echo(click.style(">> No GPUs available, using CPU", fg="red", bold=True))
    first_gpu_id = config["gpu_ids"][0]                  # 取第一块 GPU 作为主设备
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"  # 构建 torch 设备对象
    )

    # ── 2. 随机种子（保证实验可复现）────────────────────────────────────────
    if "seed" in config:
        np.random.seed(config["seed"])       # 固定 numpy 随机种子
        torch.manual_seed(config["seed"])    # 固定 PyTorch CPU 随机种子
        cudnn.deterministic = True           # cuDNN 使用确定性算法（牺牲少量速度换可复现性）
    cudnn.benchmark = True                   # 自动寻找最优卷积算法（输入尺寸固定时加速训练）

    # ── 3. 图像归一化变换（ImageNet 均值/标准差）────────────────────────────
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # 将 RGB 图像归一化到 ImageNet 分布，与预训练视觉编码器对齐
        ]
    )

    # ── 4. 加载数据集 ────────────────────────────────────────────────────────
    train_dataset = []       # 收集所有训练子数据集
    test_dataloaders = {}    # 按数据集名称存储测试 DataLoader
    for dataset_name in config["datasets"]:                    # 遍历配置中的每个数据集
        data_config = config["datasets"][dataset_name]         # 取该数据集的具体配置
        for data_split_type in ["train", "test"]:              # 分别处理训练集和测试集
            if data_split_type in data_config:                 # 若该 split 在配置中存在才构建
                dataset = ViNT_Dataset(
                    data_folder=data_config["data_folder"],            # 原始数据根目录
                    data_split_folder=data_config[data_split_type],    # train/test 的索引文件目录
                    dataset_name=dataset_name,                         # 数据集名称标识
                    image_size=config["image_size"],                   # 图像缩放尺寸
                    waypoint_spacing=data_config["waypoint_spacing"],  # 路径点采样间隔
                    min_dist_cat=config["distance"]["min_dist_cat"],   # 距离分类最小值
                    max_dist_cat=config["distance"]["max_dist_cat"],   # 距离分类最大值
                    min_action_distance=config["action"]["min_dist_cat"],  # 动作距离最小值
                    max_action_distance=config["action"]["max_dist_cat"],  # 动作距离最大值
                    negative_mining=True,                              # 启用负样本挖掘（对比学习）
                    len_traj_pred=config["len_traj_pred"],             # 预测轨迹长度（步数）
                    learn_angle=config["learn_angle"],                 # 是否学习朝向角
                    context_size=config["context_size"],               # 历史帧数（上下文窗口）
                    context_type=config["context_type"],               # 上下文类型（如 temporal）
                    end_slack=data_config["end_slack"],                # 轨迹末尾忽略的帧数
                    goals_per_obs=data_config["goals_per_obs"],        # 每个观测对应的目标数
                    normalize=config["normalize"],                     # 是否归一化动作
                    goal_type=config["goal_type"],                     # 目标类型（如 image）
                )
                if data_split_type == "train":
                    train_dataset.append(dataset)                      # 训练集追加到列表
                else:
                    dataset_type = f"{dataset_name}_{data_split_type}"
                    if dataset_type not in test_dataloaders:
                        test_dataloaders[dataset_type] = {}
                    test_dataloaders[dataset_type] = dataset           # 测试集按名称存储
    train_dataset = ConcatDataset(train_dataset)   # 将多个训练子集合并为一个大数据集
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],           # 训练批大小
        shuffle=True,                              # 每 epoch 打乱顺序
        num_workers=config["num_workers"],         # 并行数据加载进程数
        drop_last=False,                           # 保留最后一个不完整 batch
        persistent_workers=False,                  # 不持久化 worker 进程（节省内存）
    )
    click.echo(
        click.style(
            f">> Loaded {len(train_dataset)} training samples",
            fg="cyan",
            bold=True,
        )
    )
    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]   # 若未指定评估批大小，沿用训练批大小
    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset=dataset,
            batch_size=config["eval_batch_size"],  # 评估批大小
            shuffle=True,                          # 评估时也打乱，避免顺序偏差
            num_workers=0,                         # 评估时单进程加载（避免多进程开销）
            drop_last=False,
        )
        click.echo(
            click.style(
                f">> Loaded {len(dataset)} test samples for {dataset_type}",
                fg="cyan",
                bold=True,
            )
        )

    # ── 5. 构建模型 ──────────────────────────────────────────────────────────
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=config["encoding_size"],             # 视觉特征向量维度
        context_size=config["context_size"],                   # 历史帧数
        mha_num_attention_heads=config["mha_num_attention_heads"],   # 多头注意力头数
        mha_num_attention_layers=config["mha_num_attention_layers"], # Transformer 层数
        mha_ff_dim_factor=config["mha_ff_dim_factor"],         # 前馈网络维度倍数
        depth_cfg=config["depth"],                             # Depth-Anything 深度编码器配置
    )
    # 将所有 BatchNorm 替换为 GroupNorm，使多卡训练时统计量不跨卡同步，更稳定
    vision_encoder = replace_bn_with_gn(vision_encoder)

    # 条件 UNet：扩散模型的噪声预测网络，输入为带噪轨迹，条件为视觉特征
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,                                   # 轨迹维度（x, y 二维）
        global_cond_dim=config["encoding_size"],       # 全局条件维度（视觉编码）
        down_dims=config["down_dims"],                 # UNet 各下采样层的通道数
        cond_predict_scale=config["cond_predict_scale"],  # 是否用条件预测缩放因子
    )

    # 距离预测头：从视觉编码预测当前观测到目标的距离类别
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])

    # 组装完整模型：视觉编码器 + 扩散噪声预测网络 + 距离预测头
    model = NoMaD(
        vision_encoder=vision_encoder,   # 视觉编码器（历史帧 + 目标图 + 深度图 → 特征向量）
        noise_pred_net=noise_pred_net,   # 扩散噪声预测网络（生成轨迹）
        dist_pred_net=dist_pred_network, # 距离预测网络（辅助监督）
    )

    # ── 6. 优化器与学习率调度 ────────────────────────────────────────────────
    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    optimizer = AdamW(model.parameters(), lr=lr)   # AdamW 优化器，适合 Transformer 类模型

    # 余弦退火调度：学习率从 lr 余弦衰减到接近 0，共 epochs 个周期
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=config["epochs"]
    )
    # 预热调度器：前 warmup_epochs 个 epoch 线性升温，之后交给余弦退火
    scheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=1,                              # 预热结束时学习率倍数（1 = 不放大）
        total_epoch=config["warmup_epochs"],       # 预热持续的 epoch 数
        after_scheduler=scheduler,                 # 预热结束后接余弦退火
    )

    # ── 7. 断点续训（可选）──────────────────────────────────────────────────
    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        click.echo(
            click.style(
                f">> Loading pre-trained model from {load_project_folder}",
                fg="yellow",
            )
        )
        if os.path.isdir(load_project_folder):
            latest_path = os.path.join(load_project_folder, "latest.pth")  # 目录则找 latest.pth
        elif os.path.isfile(load_project_folder):
            latest_path = load_project_folder                               # 直接指定了文件路径
        else:
            click.echo(
                click.style(
                    f">> Could not find pre-trained model at {load_project_folder}",
                    fg="red",
                )
            )
        latest_checkpoint = torch.load(latest_path)                         # 加载 checkpoint 字典
        if "model" in latest_checkpoint:
            model.load_state_dict(latest_checkpoint["model"], strict=True)  # 恢复模型权重
        else:
            model.load_state_dict(latest_checkpoint, strict=True)           # checkpoint 本身就是 state_dict
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1                  # 从下一个 epoch 继续训练
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())  # 恢复优化器状态
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())  # 恢复调度器状态

    # ── 8. 加载 Depth-Anything 预训练深度编码器权重 ──────────────────────────
    checkpoint = torch.load(
        config["depth"]["weights_path"],   # 深度编码器权重文件路径
        map_location=device,               # 直接加载到目标设备，避免先加载到 CPU 再转移
    )
    # 兼容两种 checkpoint 格式：带 "state_dict" 键 或 直接是权重字典
    saved_state_dict = (
        checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    )
    # 提取键名中含 "pretrained." 的权重，并去掉该前缀，得到 backbone 权重
    updated_state_dict = {
        k.replace("pretrained.", ""): v
        for k, v in saved_state_dict.items()
        if "pretrained" in k
    }
    # 只保留与模型深度编码器结构匹配的键（过滤掉不兼容的层）
    new_state_dict = {
        k: v
        for k, v in updated_state_dict.items()
        if k in model.vision_encoder.depth_encoder.state_dict()
    }
    # 非严格加载：允许部分权重缺失（新增层随机初始化）
    model.vision_encoder.depth_encoder.load_state_dict(new_state_dict, strict=False)

    # ── 9. 多卡并行（可选）──────────────────────────────────────────────────
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])  # 数据并行，自动切分 batch 到多卡
    model = model.to(device)   # 将模型移动到主设备

    # ── 10. 启动训练主循环 ───────────────────────────────────────────────────
    main_loop(
        train_model=config["train"],           # 是否训练（False 则只评估）
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_loader=train_loader,
        test_dataloaders=test_dataloaders,
        transform=transform,                   # 图像归一化变换
        goal_mask_prob=config["goal_mask_prob"],  # 随机遮蔽目标图的概率（数据增强）
        epochs=config["epochs"],               # 总训练轮数
        device=device,
        project_folder=config["project_folder"],  # 模型/日志保存目录
        print_log_freq=config["print_log_freq"],  # 每隔多少步打印一次日志
        wandb_log_freq=config["wandb_log_freq"],  # 每隔多少步上传一次 wandb 日志
        image_log_freq=config["image_log_freq"],  # 每隔多少步记录一次可视化图像
        num_images_log=config["num_images_log"],  # 每次记录的图像数量
        current_epoch=current_epoch,           # 起始 epoch（断点续训时非 0）
        alpha=float(config["alpha"]),          # 扩散损失与距离损失的权重系数
        use_wandb=config["use_wandb"],         # 是否启用 wandb 记录
        eval_fraction=config["eval_fraction"], # 每次评估使用的测试集比例
        eval_freq=config["eval_freq"],         # 每隔多少 epoch 评估一次
    )
    click.echo(
        click.style(
            f">> Training completed. Model saved to {config['project_folder']}",
            fg="green",
            bold=True,
        )
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")  # 使用 spawn 方式启动子进程（Windows/CUDA 兼容性要求）

    # ── 命令行参数解析 ────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="config/flownav.yaml",   # 默认配置文件路径
        type=str,
        help="Path to the config file",
    )
    args = parser.parse_args()

    # ── 加载配置文件 ──────────────────────────────────────────────────────────
    this_file_dir = os.path.dirname(os.path.abspath(__file__))   # 获取本脚本所在目录（绝对路径）
    with open(f"{this_file_dir}/flownav/config/flownav.yaml", "r") as f:
        default_config = yaml.safe_load(f)   # 加载内置默认配置
    config = default_config
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)      # 加载用户指定的配置文件
    click.echo(click.style(f">> Using config file: {args.config}", fg="yellow"))

    # ── 合并配置并创建实验目录 ────────────────────────────────────────────────
    config.update(user_config)               # 用户配置覆盖默认配置（同名键以用户为准）
    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")  # 在实验名后追加时间戳，避免重名
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]   # 实验日志目录：logs/<项目名>/<实验名>
    )
    os.makedirs(
        config["project_folder"],            # 创建实验目录（若已存在会报错，保证不覆盖旧实验）
    )
    click.echo(
        click.style(
            f">> Project folder created: {config['project_folder']}", fg="yellow"
        )
    )

    # ── 初始化 wandb 实验追踪（可选）────────────────────────────────────────
    if config["use_wandb"]:
        wandb.login()                        # 登录 wandb 账号
        wandb.init(
            project=config["project_name"],  # wandb 项目名
            settings=wandb.Settings(start_method="fork"),  # 使用 fork 方式启动 wandb 后台进程
            entity=config["entity"],         # wandb 团队/用户名
        )
        wandb.save(args.config, policy="now")   # 立即上传配置文件到 wandb，便于复现
        wandb.run.name = config["run_name"]     # 设置本次运行的显示名称
        if wandb.run:
            wandb.config.update(config)         # 将完整配置上传到 wandb，记录超参数

    main(config)   # 进入主训练流程
