import torch
import yaml
from pathlib import Path
import scipy.stats as stats
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import tqdm                                          # 进度条显示
import wandb                                         # 实验追踪
from diffusers.training_utils import EMAModel        # 指数移动平均模型（稳定训练）
import numpy as np
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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLOWNAV_CONFIG_PATH = PROJECT_ROOT / "flownav" / "config" / "flownav.yaml"
with FLOWNAV_CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)             # 因为懒得改函数的接口所以把cfg的权重放在这里了
CFG_W = config["cfg_w"]                    # cfg权重
P_MEAN = config["p_mean"]                  # 动作均值
P_STD = config["p_std"]                    # 动作标准差


###### Ushape t-distribution from 2-rectified flow++
def exponential_pdf(x, a):
    C = a / (np.exp(a) - 1)  # 计算归一化常数使分布积分为 1
    return C * np.exp(a * x)  # 返回指数形式的概率密度值

# Define a custom probability density function
class ExponentialPDF(stats.rv_continuous):
    def _pdf(self, x, a):
        return exponential_pdf(x, a)  # 调用自定义密度函数供 scipy 分布对象使用

def sample_t_re2(exponential_pdf, num_samples, a):
    t = exponential_pdf.rvs(size=num_samples, a=a)  #从给定的分布采样num_samples个样本
    t = torch.from_numpy(t).float()  # 转换为 float32 的 Torch 张量
    t = torch.cat([t, 1 - t], dim=0)  # 拼接镜像样本构造 U 形分布
    t = t[torch.randperm(t.shape[0])]  # 打乱样本顺序
    t = t[:num_samples]  # 截取目标数量样本

    t_min = 1e-5  # 设定下界避免取到 0
    t_max = 1-1e-5  # 设定上界避免取到 1

    # Scale t to [t_min, t_max]
    t = t * (t_max - t_min) + t_min  # 将样本线性缩放到安全区间
    
    return t  # 返回时间采样结果

def sample_logit_normal(mean, std, shape, device):#这个函数的作用是从一个 logit-normal 分布中采样，logit-normal 分布是通过对正态分布的样本应用 sigmoid 函数得到的
        eps = torch.randn(shape, device=device)  # 采样标准高斯噪声
        x = eps * std + mean  # X ~ N(mean, std^2)
        return torch.sigmoid(x)  # Y = sigmoid(X) ∈ (0, 1)

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
                actions,           # 真实动作序列（坐标，shape: [B, T, 2]）
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
            def Mean_Flow_sample_location_and_conditional_flow(x1=naction): #代替了原来的条件流匹配器，增加了r的采样
                exponential_distribution = ExponentialPDF(a=0, b=1, name='ExponentialPDF')  # 初始化指数分布对象
                min_sigma=1e-8
                max_sigma=80/81
                t = sample_t_re2(exponential_distribution, num_samples=B, a=5.0).to(device)  # 从指数分布采样时间 t，构造 U 形分布覆盖 [0, 1]
                t = torch.clamp(t, 0, max_sigma)  # 限制到有效时间上界

                # sample interval from logit normal distribution
                interval = sample_logit_normal(P_MEAN, P_STD, (B,), device)  # 采样时间间隔
                r = t - interval  # 根据间隔计算 r
                r = torch.clamp(r, 0, 1-1e-5)  # 截断到合法区间
        
                # clamp those r with smaller than 0.4 to 0 to avoid high variance interval
                r = torch.where((t > 0.8) & (r < 0.4), torch.zeros_like(r, device=device), r)  # 抑制高噪声区间的小 r 以降低方差
                noise = torch.randn(x1.shape, device=device)
                xt=x1 * t.view(-1, 1, 1) + noise * (1 - t).view(-1, 1, 1)  # 根据采样的时间 t 插值生成噪声点 xt
                ut=(x1 - xt) / (1 - t).view(-1, 1, 1)  # 计算 xt 处的真实速度场 ut（从 xt 指向 x1 的向量）
                return t,r, xt, ut
            
            # ut 是平均速度场（x1 - x0 的方向）
            t,r, xt, ut = Mean_Flow_sample_location_and_conditional_flow(naction)
            #即：随机sample一个时间点t，在噪声点x0和真实动作点x1之间插值得到xt，ut是xt处的速度（从xt指向x1的向量，也就是x0指向x1的向量）
            #并且因为noise和naction的shape是一样的，所以t, xt, ut的shape也是一样的，都是[B, T, 2]

            # 前向：噪声预测网络（实为速度场预测），预测 xt 处的速度 vt
            with torch.no_grad():
                v_hat = CFG_W * model(
                    "noise_pred_net", sample=xt, timestep=t, global_cond=obsgoal_cond
                ) + (1 - CFG_W) * ut  # 线性插值：vt = cfg_w * model_output + (1-cfg_w) * true_velocity
                v_hat = v_hat.detach()  # 不反向传播到 v_hat 中（只更新模型参数）
             # compute jvp
            jvp_args = (  # 组织 JVP 所需函数、原点和切向量
                lambda xt, t,r:model("noise_pred_net", sample=xt, timestep=t, global_cond=obsgoal_cond),  # 被求导函数     
                (xt, t, r),  # 原点输入：在哪里求导
                (v_hat, torch.ones_like(t), torch.zeros_like(r))  # "dudt can be decomposed into two parts":即∂u/∂x * v_hat + ∂u/∂t * 1 + ∂u/∂r * 0, so the tangent vector is (v_hat, 1, 0
            )
            
            u, dudt = torch.autograd.functional.jvp(*jvp_args, create_graph=True)  # 计算前向值与沿切向量的导数并保留图
            t_r = (t - r).view(-1, 1, 1)
            u_tgt = v_hat - t_r * dudt
            error = u - u_tgt.detach()
            sq_norm = error.pow(2).mean(dim=(1, 2))          # [B]
            weight = 1.0 / (sq_norm + 1e-3).pow(2)  #误差越大权重越小，避免过大误差主导训练
            flow_loss = (weight.detach() * sq_norm).mean()
            # 总损失：距离损失（辅助）+ flow 损失（主要）
            loss = alpha * dist_loss + (1 - alpha) * flow_loss
            print(loss)
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
