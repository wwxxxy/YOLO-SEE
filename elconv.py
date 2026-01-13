import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    """
    Efficient Channel Attention (完整版)
    - 自适应确定1D卷积核大小：k = |(log2(C)/gamma + b)|，并确保为奇数
    - 全局平均池化 -> 1D卷积(沿通道维) -> Sigmoid
    参考：ECA-Net: CVPR 2020
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
        y = self.avg_pool(x)                 # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)                     # (B, 1, C)
        y = torch.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y


class ELConv(nn.Module):   #ELConv模块
    """
    - 适合 VisDrone 小目标检测
    - 调整 ECA 参数：更大感受野 (gamma=1.5, b=1.2)
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
        eca_gamma: float = 1.5,   # 更大通道卷积感受野
        eca_b: float = 1.2,
    ):
        super().__init__()
        # 基础 Depthwise 卷积
        self.dw_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        # 1x1 Pointwise 卷积
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ---- 完整版 ECA 注意力 ----
        self.eca = ECA(out_channels, k_size=eca_k_size, gamma=eca_gamma, b=eca_b)

        # ---- 局部空间增强 (LKA-lite) ----
        self.local_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1,
            groups=out_channels, bias=False
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


