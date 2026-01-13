import torch
import torch.nn as nn


class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class iRMB_MPF(nn.Module):
    def __init__(
        self, in_channels, out_channels, se_ratio=0.25, mpf_pool_sizes=[3, 5, 7], dw_ks=3, has_skip=True, act=True
    ):
        super().__init__()
        self.has_skip = (in_channels == out_channels) and has_skip

        # ---------- MPF 多尺度池化 ----------
        self.mpf = nn.ModuleList()
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
        for k in mpf_pool_sizes:
            self.mpf.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    self.act,
                    nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2),
                )
            )
        # 原始特征通过 1x1 conv
        self.mpf.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))

        # ---------- iRMB 部分 ----------
        self.norm = nn.BatchNorm2d(out_channels)
        self.conv_local = nn.Conv2d(
            out_channels, out_channels, kernel_size=dw_ks, padding=dw_ks // 2, groups=out_channels
        )
        self.se = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, int(out_channels * se_ratio), 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(out_channels * se_ratio), out_channels, 1),
                nn.Sigmoid(),
            )
            if se_ratio > 0
            else nn.Identity()
        )
        self.proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.drop_path = nn.Identity()

    def forward(self, x):
        # ---------- MPF ----------
        mpf_out = sum(branch(x) for branch in self.mpf)

        # ---------- iRMB ----------
        shortcut = mpf_out
        x = self.norm(mpf_out)
        x = self.act(x)
        x = self.conv_local(x)
        x = x * self.se(x)
        x = self.proj(x)

        if self.has_skip:
            x = shortcut + self.drop_path(x)
        return x


class C3k2iMPF(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, se_ratio=0.25):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(iRMB_MPF(self.c, self.c, se_ratio=se_ratio) for _ in range(n))
