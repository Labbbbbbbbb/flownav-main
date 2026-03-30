import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import tqdm                                          # 进度条显示
import wandb                                         # 实验追踪
from diffusers.training_utils import EMAModel        # 指数移动平均模型（稳定训练）
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher  # 条件流匹配（Flow Matching 核心）
from torchvision import transforms
from flownav.data.data_utils import VISUALIZATION_IMAGE_SIZE
from flownav.training.logger import Logger
from flownav.training.utils import (
    ACTION_STATS,                  # 动作统计量（均值/标准差），用于归一化
    action_reduce,                 # 对动作维度的损失做 mask 加权平均
    compute_losses,                # 计算详细评估指标（cos sim 等）
    get_delta,                     # 将绝对坐标动作转为相对位移（delta）
    normalize_data,                # 用统计量归一化数据到 [-1, 1]
    visualize_action_distribution, # 可视化预测轨迹分布并记录到 wandb
    from_numpy,                    # numpy → torch tensor
)


def train(
    model: nn.Module,
    ema_model: EMAModel,           # EMA 模型，用于评估和可视化（比瞬时模型更稳定）
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,         # 图像归一化变换（ImageNet 均值/标准差）
    device: torch.device,
    goal_mask_prob: float,         # 随机遮蔽目标图的概率，实现无条件/有条件混合训练
    project_folder: str,           # 模型和可视化结果的保存目录
    epoch: int,
    alpha: float = 1e-4,           # 距离损失的权重系数（flow 损失权重为 1-alpha）
    print_log_freq: int = 100,     # 每隔多少 batch 打印一次指标
    wandb_log_freq: int = 10,      # 每隔多少 batch 上传一次 wandb 日志
    image_log_freq: int = 1000,    # 每隔多少 batch 记录一次可视化图像
    num_images_log: int = 8,       # 每次可视化记录的样本数量
    use_wandb: bool = True,
):
    # 将 goal_mask_prob 裁剪到 [0, 1]，防止配置错误
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()                  # 切换到训练模式（启用 Dropout / BatchNorm 的训练行为）
    num_batches = len(dataloader)

    # ── 初始化各指标的滑动窗口 Logger ────────────────────────────────────────
    # uc = unconditional（目标被遮蔽），gc = goal-conditioned（有目标条件）
    uc_action_loss_logger = Logger(
        "uc_action_loss", "train", window_size=print_log_freq
    )
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", "train", window_size=print_log_freq)
    gc_action_loss_logger = Logger(
        "gc_action_loss", "train", window_size=print_log_freq
    )
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }

    # ── 遍历所有 batch（带进度条）────────────────────────────────────────────
    with tqdm.tqdm(
        dataloader,
        desc=f"Train epoch {epoch}",
        leave=True,
        dynamic_ncols=True,        # 自动适应终端宽度
        colour="magenta",
    ) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image,         # 观测图像（多帧拼接在通道维度，shape: [B, 3*context_size, H, W]）
                goal_image,        # 目标图像（shape: [B, 3, H, W]）
                actions,           # 真实动作序列（绝对坐标，shape: [B, T, 2]）
                distance,          # 到目标的距离标签（标量）
                goal_pos,          # 目标位置（用于可视化）
                _,                 # 占位符（数据集返回的额外字段，此处不用）
                action_mask,       # 动作有效掩码（某些 step 可能无效）
            ) = data

            # 将多帧拼接的观测图像按 3 通道拆分为单帧列表
            obs_images = torch.split(obs_image, 3, dim=1)
            # 取最后一帧（最新观测）缩放到可视化尺寸，用于后续记录图像
            batch_viz_obs_images = TF.resize(
                obs_images[-1], VISUALIZATION_IMAGE_SIZE[::-1]
            )
            # 目标图像也缩放到可视化尺寸
            batch_viz_goal_images = TF.resize(
                goal_image, VISUALIZATION_IMAGE_SIZE[::-1]
            )
            # 对每帧观测图像做 ImageNet 归一化，再沿通道维度拼接，移到 GPU
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            # 目标图像归一化并移到 GPU
            batch_goal_images = transform(goal_image).to(device)

            # 动作掩码移到 GPU
            action_mask = action_mask.to(device)

            # 将绝对坐标动作转为相邻帧间的相对位移（delta），再归一化到 [-1, 1]  （处理动作数据，相当于labels）
            deltas = get_delta(actions)
            ndeltas = normalize_data(deltas, ACTION_STATS)
            naction = from_numpy(ndeltas).to(device)   # 转为 tensor 并移到 GPU

            B = actions.shape[0]   # batch size

            # 随机生成 goal mask：1 表示遮蔽目标（无条件），0 表示保留目标（有条件）
            goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)

            # 前向：视觉编码器，融合历史观测 + 目标图（部分被遮蔽），输出条件特征向量
            obsgoal_cond = model(
                "vision_encoder",
                obs_img=batch_obs_images,
                goal_img=batch_goal_images,
                input_goal_mask=goal_mask,
            )

            distance = distance.float().to(device)          #temporal distance 的 labels

            # 前向：距离预测头，从条件特征预测到目标的距离
            dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)   #要将前面的transformer block（vision encoder）的输出作为distance predictor的输入
            # MSE 距离损失，只在有目标条件（goal_mask=0）的样本上计算
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)
            dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / (
                1e-2 + (1 - goal_mask.float()).mean()  # 归一化，避免除零
            )

            # ── Flow Matching 训练 ────────────────────────────────────────────
            # 从标准正态分布采样噪声作为流的起点 x0
            noise = torch.randn(naction.shape, device=device)

            # 构建条件流匹配器（sigma=0 表示确定性 ODE 流，无随机性）
            FM = ConditionalFlowMatcher(sigma=0.0)
            # 在 x0（噪声）和 x1（真实动作）之间采样时间 t 和插值点 xt，
            # ut 是该点处的真实速度场（x1 - x0 的方向）
            t, xt, ut = FM.sample_location_and_conditional_flow(x0=noise, x1=naction)
            #即：随机sample一个时间点t，在噪声点x0和真实动作点x1之间插值得到xt，ut是xt处的速度（从xt指向x1的向量，也就是x0指向x1的向量）
            #并且因为noise和naction的shape是一样的，所以t, xt, ut的shape也是一样的，都是[B, T, 2]

            # 前向：噪声预测网络（实为速度场预测），预测 xt 处的速度 vt
            vt = model(
                "noise_pred_net", sample=xt, timestep=t, global_cond=obsgoal_cond
            )

            # Flow 损失：预测速度场 vt 与真实速度场 ut 的 L2 距离，用 action_mask 加权
            flow_loss = action_reduce(F.mse_loss(vt, ut, reduction="none"), action_mask)

            # 总损失：距离损失（辅助）+ flow 损失（主要）
            loss = alpha * dist_loss + (1 - alpha) * flow_loss

            # ── 反向传播与参数更新 ────────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新 EMA 权重（对模型参数做指数移动平均，用于评估）
            ema_model.step(model)

            # ── 日志记录 ──────────────────────────────────────────────────────
            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)   # 在进度条右侧显示当前 loss
            if use_wandb:
                wandb.log({"total_loss": loss_cpu})
                wandb.log({"dist_loss": dist_loss.item()})
                wandb.log({"flow_loss": flow_loss.item()})

            # 每 print_log_freq 个 batch 计算详细指标（cos sim 等）并打印
            if i % print_log_freq == 0:
                losses = compute_losses(
                    ema_model=ema_model.averaged_model,    # 用 EMA 模型评估，更稳定
                    batch_obs_images=batch_obs_images,
                    batch_goal_images=batch_goal_images,
                    batch_dist_label=distance.to(device),
                    batch_action_label=actions.to(device),
                    device=device,
                    action_mask=action_mask.to(device),
                    use_wandb=use_wandb,
                )

                # 将各指标值写入对应的滑动窗口 Logger
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())

                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()   # 取最新滑动均值
                    if i % print_log_freq == 0 and print_log_freq != 0:
                        print(
                            f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}"
                        )

                # 按 wandb_log_freq 频率上传详细指标
                if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
                    wandb.log(data_log, commit=True)

            # 按 image_log_freq 频率可视化预测轨迹分布并保存/上传
            if image_log_freq != 0 and i % image_log_freq == 0:
                visualize_action_distribution(
                    ema_model=ema_model.averaged_model,
                    batch_obs_images=batch_obs_images,
                    batch_goal_images=batch_goal_images,
                    batch_viz_obs_images=batch_viz_obs_images,
                    batch_viz_goal_images=batch_viz_goal_images,
                    batch_action_label=actions,
                    batch_distance_labels=distance,
                    batch_goal_pos=goal_pos,
                    device=device,
                    eval_type="train",
                    project_folder=project_folder,
                    epoch=epoch,
                    num_images_log=num_images_log,
                    num_samples=4,                 # 每个样本采样 4 条轨迹用于可视化
                    use_wandb=use_wandb,
                )
