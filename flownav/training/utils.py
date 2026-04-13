import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq
import wandb
import yaml
from diffusers.training_utils import EMAModel
from flownav.visualizing.plot import plot_trajs_and_points


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()


def load_data_stats() -> dict:
    with open(
        os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r"
    ) as f:
        data_config = yaml.safe_load(f)
    action_stats = {}
    for key in data_config["action_stats"]:
        action_stats[key] = np.array(data_config["action_stats"][key])
    return action_stats


ACTION_STATS = load_data_stats()


def action_reduce(
    unreduced_loss: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
    # Reduce over non-batch dimensions to get loss per batch element
    while unreduced_loss.dim() > 1:
        unreduced_loss = unreduced_loss.mean(dim=-1)
    assert unreduced_loss.shape == action_mask.shape, (
        f"{unreduced_loss.shape} != {action_mask.shape}"
    )
    return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)


def compute_losses(
    ema_model: EMAModel,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_dist_label: torch.Tensor,
    batch_action_label: torch.Tensor,
    device: torch.device,
    action_mask: torch.Tensor,
    use_wandb: bool,
) -> dict[str, torch.Tensor]:
    # Get pred horizon and action dimension
    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    # Get model output
    model_output_dict = model_output(
        model=ema_model,
        batch_obs_images=batch_obs_images,
        batch_goal_images=batch_goal_images,
        pred_horizon=pred_horizon,
        action_dim=action_dim,
        num_samples=1,
        device=device,
        use_wandb=use_wandb,
    )
    uc_actions = model_output_dict["uc_actions"]
    gc_actions = model_output_dict["gc_actions"]
    gc_distance = model_output_dict["gc_distance"]

    # Compute losses
    gc_dist_loss = F.mse_loss(gc_distance, batch_dist_label.unsqueeze(-1))

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert uc_actions.shape == batch_action_label.shape, (
        f"{uc_actions.shape} != {batch_action_label.shape}"
    )
    assert gc_actions.shape == batch_action_label.shape, (
        f"{gc_actions.shape} != {batch_action_label.shape}"
    )

    # Compute action losses
    uc_action_loss = action_reduce(
        F.mse_loss(uc_actions, batch_action_label, reduction="none"), action_mask
    )
    gc_action_loss = action_reduce(
        F.mse_loss(gc_actions, batch_action_label, reduction="none"), action_mask
    )

    uc_action_waypts_cos_similairity = action_reduce(
        F.cosine_similarity(uc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1),
        action_mask,
    )
    uc_multi_action_waypts_cos_sim = action_reduce(
        F.cosine_similarity(
            torch.flatten(uc_actions[:, :, :2], start_dim=1),
            torch.flatten(batch_action_label[:, :, :2], start_dim=1),
            dim=-1,
        ),
        action_mask,
    )
    gc_action_waypts_cos_similairity = action_reduce(
        F.cosine_similarity(gc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1),
        action_mask,
    )
    gc_multi_action_waypts_cos_sim = action_reduce(
        F.cosine_similarity(
            torch.flatten(gc_actions[:, :, :2], start_dim=1),
            torch.flatten(batch_action_label[:, :, :2], start_dim=1),
            dim=-1,
        ),
        action_mask,
    )

    results = {
        "uc_action_loss": uc_action_loss,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_similairity,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim,
        "gc_dist_loss": gc_dist_loss,
        "gc_action_loss": gc_action_loss,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_similairity,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim,
    }

    return results


def normalize_data(data: np.ndarray, stats: dict) -> np.ndarray:
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats) -> np.ndarray:
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    return data


def get_delta(actions: np.ndarray) -> np.ndarray:
    ex_actions = np.concatenate(
        [np.zeros((actions.shape[0], 1, actions.shape[-1])), actions], axis=1
    )
    delta = ex_actions[:, 1:] - ex_actions[:, :-1]
    return delta


def get_action(ndeltas, action_stats=ACTION_STATS) -> torch.Tensor:
    device = ndeltas.device
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(ndeltas, axis=1)
    return from_numpy(actions).to(device)


def model_output(
    model: nn.Module,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
    use_wandb: bool,
) -> dict[str, torch.Tensor]:
    # Exploration
    goal_mask = torch.ones((batch_goal_images.shape[0],)).long().to(device)
    obs_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_img=batch_goal_images,
        input_goal_mask=goal_mask,
    )
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

    # Navigation
    no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
    obsgoal_cond = model(
        "vision_encoder",
        obs_img=batch_obs_images,
        goal_img=batch_goal_images,
        input_goal_mask=no_mask,
    )
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    with torch.no_grad():
        start_time = time.time()

        # Exploration
        output = torch.randn((len(obs_cond), pred_horizon, action_dim), device=device)
        traj = torchdiffeq.odeint(
            lambda t, x: model.forward(
                "noise_pred_net", sample=x, timestep=t, global_cond=obs_cond
            ),
            output,
            torch.linspace(0, 1, 10, device=device),
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        uc_actions = get_action(traj[-1], ACTION_STATS)
        proc_time = time.time() - start_time
        mean_proc_time = proc_time / output.shape[0]
        if use_wandb:
            wandb.log({"Mean Processing Time UC": mean_proc_time})
            wandb.log({"Processing Time UC": proc_time})

        # Navigation
        output = torch.randn((len(obs_cond), pred_horizon, action_dim), device=device)
        traj = torchdiffeq.odeint(
            lambda t, x: model.forward(
                "noise_pred_net", sample=x, timestep=t, global_cond=obsgoal_cond
            ),
            output,
            torch.linspace(0, 1, 10, device=device),
            atol=1e-4,
            rtol=1e-4,
            method="euler",
        )
        gc_actions = get_action(traj[-1], ACTION_STATS)
        proc_time = time.time() - start_time
        mean_proc_time = proc_time / output.shape[0]
        if use_wandb:
            wandb.log({"Mean Processing Time GC": mean_proc_time})
            wandb.log({"Processing Time GC": proc_time})

    # Predict distance
    obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
    gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

    return {
        "uc_actions": uc_actions,
        "gc_actions": gc_actions,
        "gc_distance": gc_distance,
    }


def visualize_action_distribution(
    ema_model: nn.Module,           # EMA 平均后的模型，用于推理（比瞬时模型更稳定）
    batch_obs_images: torch.Tensor, # 经过 transform 的观测图像，用于模型输入，shape (B, C*context, H, W)
    batch_goal_images: torch.Tensor,# 经过 transform 的目标图像，用于模型输入，shape (B, C, H, W)
    batch_viz_obs_images: torch.Tensor,  # 原始尺寸观测图像，仅用于可视化展示，不送入模型
    batch_viz_goal_images: torch.Tensor, # 原始尺寸目标图像，仅用于可视化展示，不送入模型
    batch_action_label: torch.Tensor,    # ground truth 动作轨迹，shape (B, pred_horizon, action_dim)
    batch_distance_labels: torch.Tensor, # ground truth 距离标签（帧数），shape (B,)
    batch_goal_pos: torch.Tensor,        # ground truth 目标位置（2D 坐标），shape (B, 2)，用于在轨迹图上标点
    device: torch.device,
    eval_type: str,          # 评估类型字符串（如 "train"/"val"），用于区分保存路径和 wandb key
    project_folder: str,     # 项目根目录，可视化图片保存在其下的 visualize/ 子目录
    epoch: int,              # 当前 epoch，用于构建保存路径
    num_images_log: int,     # 最多可视化的样本数量上限
    num_samples: int = 30,   # 每个样本生成的预测轨迹条数（越多分布越完整，但推理越慢）
    use_wandb: bool = True,  # 是否将图片上传到 Weights & Biases
) -> None:
    # 构建本次可视化的保存目录路径：project_folder/visualize/{eval_type}/epoch{epoch}/action_sampling_prediction/
    
    print('ema_model.training:',ema_model.training)





    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    # 若目录不存在则递归创建
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    # 记录原始 batch 大小，后续 torch.split 时用作每块的最大尺寸（此处等于整个 batch，实际不做切分）
    max_batch_size = batch_obs_images.shape[0]
    # 实际可视化数量取 num_images_log 与各 tensor 第0维的最小值，防止越界
    num_images_log = min(
        num_images_log,
        batch_obs_images.shape[0],
        batch_goal_images.shape[0],
        batch_action_label.shape[0],
        batch_goal_pos.shape[0],
    )
    # 截取前 num_images_log 个样本，丢弃多余的
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_goal_images = batch_goal_images[:num_images_log]
    batch_action_label = batch_action_label[:num_images_log]
    batch_goal_pos = batch_goal_pos[:num_images_log]

    # 初始化 wandb 上传列表
    wandb_list = []
    # 从 ground truth 动作标签中读取预测时域长度和动作维度，传给 model_output 用于分配输出 tensor
    pred_horizon = batch_action_label.shape[1]  # 预测步数，如 8
    action_dim = batch_action_label.shape[2]    # 动作维度，如 2（x, y）

    # 将 batch 按 max_batch_size 切分为子块列表（此处 max_batch_size == num_images_log，只有一块）
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)
    # 用于收集各子块的推理结果
    uc_actions_list = []   # 无条件（探索模式）预测轨迹
    gc_actions_list = []   # 有目标条件（导航模式）预测轨迹
    gc_distances_list = [] # 有目标条件下预测的距离值

    # 遍历各子块，调用 model_output 进行推理
    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        model_output_dict = model_output(
            model=ema_model,
            batch_obs_images=obs,
            batch_goal_images=goal,
            pred_horizon=pred_horizon,
            action_dim=action_dim,
            num_samples=num_samples,  # 每个样本生成 num_samples 条轨迹
            device=device,
            use_wandb=use_wandb,
        )
        # 将 GPU tensor 转为 numpy，便于后续 matplotlib 绘图
        uc_actions_list.append(to_numpy(model_output_dict["uc_actions"]))
        gc_actions_list.append(to_numpy(model_output_dict["gc_actions"]))
        gc_distances_list.append(to_numpy(model_output_dict["gc_distance"]))

    # 将各子块结果沿 axis=0 拼接，得到完整的预测结果数组
    # shape: (num_images_log * num_samples, pred_horizon, action_dim)
    uc_actions_list = np.concatenate(uc_actions_list, axis=0)
    gc_actions_list = np.concatenate(gc_actions_list, axis=0)
    gc_distances_list = np.concatenate(gc_distances_list, axis=0)

    # 将拼接后的数组按样本数切分，每个元素对应一个样本的 num_samples 条轨迹
    # 切分后每个元素 shape: (num_samples, pred_horizon, action_dim)
    uc_actions_list = np.split(uc_actions_list, num_images_log, axis=0)
    gc_actions_list = np.split(gc_actions_list, num_images_log, axis=0)
    gc_distances_list = np.split(gc_distances_list, num_images_log, axis=0)
    # 计算每个样本的距离预测均值和标准差，用于图标题显示
    gc_distances_avg = [np.mean(dist) for dist in gc_distances_list]
    gc_distances_std = [np.std(dist) for dist in gc_distances_list]
    # 将 ground truth 距离标签转为 numpy，用于图标题显示
    np_distance_labels = to_numpy(batch_distance_labels)

    # 逐样本绘图
    for i in range(num_images_log):
        # 创建 1 行 3 列的子图：左=轨迹分布，中=观测图，右=目标图
        fig, ax = plt.subplots(1, 3)
        uc_actions = uc_actions_list[i]  # 第 i 个样本的无条件预测轨迹，shape (num_samples, pred_horizon, 2)
        gc_actions = gc_actions_list[i]  # 第 i 个样本的有条件预测轨迹，shape (num_samples, pred_horizon, 2)
        action_label = to_numpy(batch_action_label[i])  # ground truth 轨迹，shape (pred_horizon, 2)
        # 将三类轨迹拼接为一个数组，方便统一传给绘图函数
        # action_label[None] 增加一个维度使其变为 (1, pred_horizon, 2)，与前两者对齐
        traj_list = np.concatenate(
            [
                uc_actions,       # 红色：无条件探索轨迹（num_samples 条）
                gc_actions,       # 绿色：有目标导航轨迹（num_samples 条）
                action_label[None],  # 品红：ground truth 轨迹（1 条）
            ],
            axis=0,
        )
        # 为每条轨迹指定颜色：探索=红，导航=绿，ground truth=品红
        traj_colors = (
            ["red"] * len(uc_actions) + ["green"] * len(gc_actions) + ["magenta"]
        )
        # 预测轨迹半透明（alpha=0.1）以展示分布，ground truth 不透明（alpha=1.0）突出显示
        traj_alphas = [0.1] * (len(uc_actions) + len(gc_actions)) + [1.0]
        # 在轨迹图上标两个点：机器人当前位置 (0,0) 和 ground truth 目标位置
        point_list = [np.array([0, 0]), to_numpy(batch_goal_pos[i])]
        point_colors = ["green", "red"]  # 当前位置=绿，目标位置=红
        point_alphas = [1.0, 1.0]
        # 绘制所有轨迹和标记点到左侧子图
        plot_trajs_and_points(
            ax=ax[0],
            list_trajs=traj_list,
            list_points=point_list,
            traj_colors=traj_colors,
            point_colors=point_colors,
            traj_labels=None,
            point_labels=None,
            quiver_freq=0,       # 不绘制方向箭头
            traj_alphas=traj_alphas,
            point_alphas=point_alphas,
        )
        # 将观测和目标图像从 (C, H, W) 转为 (H, W, C)，imshow 需要 channel-last 格式
        obs_image = to_numpy(batch_viz_obs_images[i])
        goal_image = to_numpy(batch_viz_goal_images[i])
        obs_image = np.moveaxis(obs_image, 0, -1)   # (C,H,W) → (H,W,C)
        goal_image = np.moveaxis(goal_image, 0, -1) # (C,H,W) → (H,W,C)
        ax[1].imshow(obs_image)   # 中图：当前观测帧
        ax[2].imshow(goal_image)  # 右图：目标图像
        ax[0].set_title("action predictions")
        ax[1].set_title("observation")
        # 右图标题同时显示 ground truth 距离标签和模型预测距离（均值±标准差）
        ax[2].set_title(
            f"goal: label={np_distance_labels[i]} gc_dist={gc_distances_avg[i]:.2f}±{gc_distances_std[i]:.2f}"
        )

        # 放大图片尺寸，使三个子图都清晰可读
        fig.set_size_inches(18.5, 10.5)
        # 保存到磁盘：project_folder/visualize/{eval_type}/epoch{epoch}/action_sampling_prediction/sample_{i}.png
        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        # 将保存的图片包装为 wandb.Image 对象，加入上传列表
        wandb_list.append(wandb.Image(save_path))
        # 关闭当前 figure，释放内存，避免多次循环后内存泄漏
        plt.close(fig)

    # 若有图片且启用 wandb，则批量上传所有可视化图片
    # commit=False 表示不立即提交 wandb step，等待后续其他 log 一起提交
    if len(wandb_list) > 0 and use_wandb:
        wandb.log({f"{eval_type}_action_samples": wandb_list}, commit=False)
