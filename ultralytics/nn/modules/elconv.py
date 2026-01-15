from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    """Efficient Channel Attention (完整版) - 自适应确定1D卷积核大小：k = |(log2(C)/gamma + b)|，并确保为奇数 - 全局平均池化 -> 1D卷积(沿通道维) ->
    Sigmoid 参考：ECA-Net: CVPR 2020.
    """

    def __init__(self, channels: int, k_size: int | None = None, gamma: float = 2.0, b: float = 1.0):
        super().__init__()
        if k_size is None:
            # 自适应核大小（并保证为奇数且至少为1）
            k = int(abs((math.log2(channels) / gamma) + b))
            k = max(1, k)
            if k % 2 == 0:
                k += 1
            k_size = k
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = torch.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y


class ELConv(nn.Module):  # ELConv模块
    """- 适合 VisDrone 小目标检测 - 调整 ECA 参数：更大感受野 (gamma=1.5, b=1.2).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        # VisDrone 适配参数
        eca_k_size: int | None = None,
        eca_gamma: float = 1.5,  # 更大通道卷积感受野
        eca_b: float = 1.2,
    ):
        super().__init__()
        # 基础 Depthwise 卷积
        self.dw_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        # 1x1 Pointwise 卷积
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ---- 完整版 ECA 注意力 ----
        self.eca = ECA(out_channels, k_size=eca_k_size, gamma=eca_gamma, b=eca_b)

        # ---- 局部空间增强 (LKA-lite) ----
        self.local_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False
        )
        self.local_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise
        out = self.dw_conv(x)
        out = self.bn1(out)
        out = F.silu(out, inplace=True)

        # Pointwise
        out = self.pw_conv(out)
        out = self.bn2(out)
        out = F.silu(out, inplace=True)

        # 完整版 ECA 注意力
        out = self.eca(out)

        # 局部空间增强（残差）
        out_local = self.local_conv(out)
        out_local = self.local_bn(out_local)
        out = out + out_local

        return out


import torch.nn as nn
from einops import rearrange

# cvpr 2024
# https://arxiv.org/pdf/2311.11587
# https://github.com/CV-ZhangXin/LDConv


class LDConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super().__init__()
        self.num_param = num_param
        self.stride = stride
        self.ldconv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
            nn.BatchNorm2d(outc),
            nn.SiLU(),
        )  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat(
            [torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)], dim=-1
        ).long()
        q_rb = torch.cat(
            [torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)], dim=-1
        ).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = (
            g_lt.unsqueeze(dim=1) * x_q_lt
            + g_rb.unsqueeze(dim=1) * x_q_rb
            + g_lb.unsqueeze(dim=1) * x_q_lb
            + g_rt.unsqueeze(dim=1) * x_q_rt
        )

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.ldconv(x_offset)

        return out

    # generating the initial sampled shapes for the LDConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(torch.arange(0, row_number), torch.arange(0, base_int), indexing="ij")
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1), torch.arange(0, mod_number), indexing="ij"
            )

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride), torch.arange(0, w * self.stride, self.stride), indexing="ij"
        )

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        _b, _c, _h, _w, _n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)

        x_offset = rearrange(x_offset, "b c h w n -> b c (h n) w")
        return x_offset


# class LSConv(nn.Module):
#     """
#     改进版 LSConv 用于替换 YOLO11 Detect 头部的 cv2
#     特点：
#     - DWConv + PWConv 基础卷积
#     - 轻量化通道注意力 (ECA)
#     - 局部空间卷积增强 (3x3 depthwise)
#     """
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, reduction=4):
#         super().__init__()
#         # 基础DW卷积
#         self.dw_conv = nn.Conv2d(
#             in_channels, in_channels, kernel_size=kernel_size,
#             stride=stride, padding=padding, groups=in_channels, bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         # 1x1 pointwise卷积
#         self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#         # ---- 改进点1：轻量ECA注意力 (取代SE) ----
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         self.eca = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)

#         # ---- 改进点2：局部空间增强 (LKA-lite) ----
#         self.local_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
#                                     groups=out_channels, bias=False)
#         self.local_bn = nn.BatchNorm2d(out_channels)

#     def forward(self, x):
#         # Depthwise
#         out = self.dw_conv(x)
#         out = self.bn1(out)
#         out = F.silu(out, inplace=True)

#         # Pointwise
#         out = self.pw_conv(out)
#         out = self.bn2(out)
#         out = F.silu(out, inplace=True)

#         # ECA通道注意力
#         w = self.global_pool(out)  # (B, C, 1, 1)
#         w = w.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
#         w = self.eca(w)
#         w = torch.sigmoid(w).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
#         out = out * w

#         # 局部空间增强
#         out_local = self.local_conv(out)
#         out_local = self.local_bn(out_local)
#         out = out + out_local  # 残差增强

#         return out


# class LSConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, reduction=4):
#         super().__init__()
#         self.dw_conv = nn.Conv2d(
#             in_channels, in_channels, kernel_size=kernel_size, stride=stride,
#             padding=padding, groups=in_channels, bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#         # 局部注意机制
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(out_channels, out_channels // reduction, 1, bias=False)
#         self.fc2 = nn.Conv2d(out_channels // reduction, out_channels, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # Depthwise卷积
#         out = self.dw_conv(x)
#         out = self.bn1(out)
#         out = F.relu(out, inplace=True)

#         # Pointwise卷积
#         out = self.pw_conv(out)
#         out = self.bn2(out)
#         out = F.relu(out, inplace=True)

#         # 局部注意机制
#         w = self.global_pool(out)
#         w = self.fc1(w)
#         w = F.relu(w, inplace=True)
#         w = self.fc2(w)
#         w = self.sigmoid(w)

#         out = out * w
#         return out
