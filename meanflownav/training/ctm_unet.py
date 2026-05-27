from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

from consistency_policy.diffusion_unet_with_dropout import ConditionalResidualBlock1D 

logger = logging.getLogger(__name__)




class CTMConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        dropout_rate=.0,
        use_v_head=False,
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        self.dropout_rate = dropout_rate

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed * 2 # for both timestep and stoptime
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, dropout_rate=dropout_rate),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, dropout_rate=dropout_rate),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))



        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, dropout_rate=dropout_rate),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, dropout_rate=dropout_rate),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))


        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv
        self.use_v_head = use_v_head
        if use_v_head:
            self.v_head_conv = nn.Sequential(
                Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
                nn.Conv1d(start_dim, input_dim, 1),
            )

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )


    def prepare_drop_generators(self):
        dropout_generator = torch.Generator().manual_seed(42)
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.generator = dropout_generator
                if self.dropout_rate == 0.0:
                    module.generator = None

    def _encode(self, sample, timestep, stoptime, local_cond=None, global_cond=None):
        """共享的 backbone，返回最后一层特征（final_conv 之前）"""
        sample = einops.rearrange(sample, 'b h t -> b t h')
        
        encoded_times = self.diffusion_step_encoder(timestep)
        encoded_stops = self.diffusion_step_encoder(stoptime)
        global_feature = torch.cat([encoded_times, encoded_stops], axis=-1)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        h_local = []
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            h_local.append(resnet(local_cond, global_feature))
            h_local.append(resnet2(local_cond, global_feature))

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        return x  # (B, start_dim, T)，final_conv 之前的特征

    def forward(self, sample, timestep, stoptime, local_cond=None, global_cond=None, **kwargs):
        """u 预测头"""
        x = self._encode(sample, timestep, stoptime, local_cond, global_cond)
        x = self.final_conv(x)
        return einops.rearrange(x, 'b t h -> b h t')

    def forward_v(self, sample, timestep, local_cond=None, global_cond=None):
        """v 预测头：stoptime=timestep，即 h=0 的瞬时速度"""
        assert self.use_v_head, "需要初始化时设置 use_v_head=True"
        x = self._encode(sample, timestep, timestep, local_cond, global_cond)
        x = self.v_head_conv(x)
        return einops.rearrange(x, 'b t h -> b h t')

    