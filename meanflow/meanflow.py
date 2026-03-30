import torch
import torch.nn as nn
from functools import partial


class MeanFlowTrajectory:
    """
    MeanFlow 框架，适配轨迹生成（替换图像的 4D 操作为轨迹的 3D 操作）。
    
    teacher_model: 已训练好的 FlowNav noise_pred_net（ConditionalUnet1D），
                   作为 1-rectified flow 提供 v_hat
    student_model: 新的网络，输入 (xr, t, r, cond)，输出平均速度 u
    """
    def __init__(self, teacher_model, student_model, consistency_ratio=0.25, pc=1e-3, p=2.0):
        self.teacher = teacher_model   # 冻结，只做推理
        self.student = student_model   # 训练目标
        self.consistency_ratio = consistency_ratio
        self.pc = pc
        self.p = p

    def get_xt(self, x0, noise, t):
        # x0: [B, T, 2], noise: [B, T, 2], t: [B]
        t = t.view(-1, 1, 1)           # 广播到轨迹维度
        return (1 - t) * x0 + t * noise

    def sample_t_and_r(self, bs, device):
        t = torch.rand(bs, device=device)
        r = t * torch.rand(bs, device=device)   # r ∈ [0, t)
        # consistency_ratio 比例的样本令 r=t（一致性路径）
        mask = torch.rand(bs, device=device) < (1 - self.consistency_ratio)
        r[mask] = t[mask]
        return t, r, mask

    def get_loss(self, error):
        # error: [B, T, 2]，自适应权重损失
        sq_norm = error.pow(2).mean(dim=(1, 2))          # [B]
        weight = 1.0 / (sq_norm + self.pc).pow(self.p)  #误差越大权重越小，避免过大误差主导训练
        return (weight.detach() * sq_norm).mean()

    def loss(self, x0, noise, obsgoal_cond):
        """
        x0: 归一化后的真实轨迹 naction [B, T, 2]
        noise: 采样的高斯噪声 [B, T, 2]
        obsgoal_cond: 视觉编码器输出的条件特征 [B, D]
        """
        bs = x0.shape[0]
        device = x0.device

        t, r, _ = self.sample_t_and_r(bs, device)

        # 1-rectified flow 的瞬时速度目标（教师模型，冻结）
        xt = self.get_xt(x0, noise, t)
        with torch.no_grad():
            v_hat = self.teacher(   #teacher_model 只做推理，输入 xt、t 和 obsgoal_cond，输出 v_hat
                sample=xt,
                timestep=t,
                global_cond=obsgoal_cond
            ).detach()

        # 学生模型：预测从 xr 出发到 x0 的平均速度
        xr = self.get_xt(x0, noise, r)

        # 用 JVP 计算 du/dt（沿时间方向的导数）
        def student_fn(xr_, t_, r_):
            return self.student(sample=xr_, timestep=t_, global_cond=obsgoal_cond, r=r_)

        u, dudt = torch.autograd.functional.jvp(
            student_fn,
            (xr, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),  # 切向量：dx=v_hat, dt=1, dr=0
            create_graph=True,
        )

        # MeanFlow 目标：u_tgt = v_hat - (t-r) * du/dt
        t_r = (t - r).view(-1, 1, 1)
        u_tgt = v_hat - t_r * dudt

        error = u - u_tgt.detach()
        return self.get_loss(error)


'''
ConditionalUnet1D 的 forward 签名是 (sample, timestep, global_cond)，MeanFlow 需要额外传入 r。你需要一个包装器或新模型，把 r 也编码进去
注意：这要求在 train.py 里把 ConditionalUnet1D 的 global_cond_dim 改为 encoding_size + 1。
'''
class MeanFlowUnet(nn.Module):
    """
    在 ConditionalUnet1D 基础上，将 r 也编码到时间嵌入中。
    """
    def __init__(self, unet):
        super().__init__()
        self.unet = unet   # ConditionalUnet1D

    def forward(self, sample, timestep, global_cond, r):
        # 最简单的做法：把 r 拼到 global_cond 里
        # 或者：把 (t, r) 都编码为时间嵌入（需要改 UNet 内部）
        # 这里用最简单的方案：将 r 作为额外标量拼到 global_cond
        r_emb = r.unsqueeze(-1)                          # [B, 1]
        cond = torch.cat([global_cond, r_emb], dim=-1)   # [B, D+1]
        return self.unet(sample, timestep, cond)



'''

# 原来的 Flow Matching（保留教师模型用）
# FM = ConditionalFlowMatcher(sigma=0.0)
# t, xt, ut = FM.sample_location_and_conditional_flow(x0=noise, x1=naction)
# vt = model("noise_pred_net", ...)
# flow_loss = F.mse_loss(vt, ut)

# 替换为 MeanFlow loss
flow_loss = meanflow.loss(
    x0=naction,
    noise=noise,
    obsgoal_cond=obsgoal_cond,
)


问题	说明
教师模型冻结	teacher_model 加 model.eval() + torch.no_grad()，不参与梯度更新
JVP 的切向量	(v_hat, 1, 0) 对应论文中 du/dt = ∂u/∂x·v + ∂u/∂t·1 + ∂u/∂r·0
维度适配	原版 MeanFlow 用 mean(dim=(1,2,3))，轨迹改为 mean(dim=(1,2))
r 的注入方式	最简单是拼到 global_cond；更好的是修改 UNet 的时间嵌入层同时接受 t 和 r
推理时	一步采样：x0 = xr - student(xr, t=1, r=0, cond)，不再需要多步 ODE

'''