import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == "relu":
        layer = nn.ReLU(inplace)
    elif act == "relu6":
        layer = nn.ReLU6(inplace)
    elif act == "leakyrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == "gelu":
        layer = nn.GELU()
    elif act == "hswish":
        layer = nn.Hardswish(inplace)
    elif act == "silu" or act == "swish":
        layer = nn.SiLU(inplace=inplace)
    else:
        raise NotImplementedError(f"activation layer [{act}] is not found")
    return layer


class SEUCB(nn.Module):  # EUCBS模块
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        activation="silu",
        shuffle_groups=2,
        use_residual=True,
        up_mode="bilinear",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shuffle_groups = shuffle_groups
        self.use_residual = use_residual and (in_channels == out_channels)

        # 上采样 + DWConv
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=up_mode, align_corners=False if up_mode == "bilinear" else None),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            act_layer(activation, inplace=True),
        )

        # Pointwise Conv (1x1) + BN + Act
        self.pwc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            act_layer(activation, inplace=True),
        )

    def forward(self, x):
        identity = x

        x = self.up_dwc(x)
        x = channel_shuffle(x, self.shuffle_groups)
        x = self.pwc(x)

        if self.use_residual:
            # 上采样残差：最近邻插值对 identity 上采样再加
            identity = F.interpolate(identity, scale_factor=2, mode="nearest")
            x = x + identity

        return x
