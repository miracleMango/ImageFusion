import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.contrib.models.common import DropPath

from models.debug import print_channel_distribution


class SEAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化：(B,C,H,W)→(B,C,1,1)，捕捉通道全局统计
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),  # 降维，减少计算
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),  # 升维，输出通道权重
            nn.Sigmoid()  # 权重归一化到[0,1]，表示每个通道的重要性
        )
        # 新增：调试标记（关联全局step）
        self.debug_step = 0

    def forward(self, x):
        b, c, h, w = x.shape
        # 1. 全局平均池化：汇总每个通道的空间信息，得到通道级统计特征
        channel_stats = self.gap(x).view(b, c)  # (B,C)
        # 2. 学习通道权重
        channel_weights = self.fc(channel_stats).view(b, c, 1, 1)  # (B,C,1,1)

        # ========== 新增：调试通道权重 ==========
        if hasattr(self, 'debug_step') and self.debug_step < 5:  # 只打印前5步
            print(f"\n----- SEAttention 通道权重 (Step {self.debug_step}) -----")
            # 若特征通道数≥3，假设前3通道对应RGB
            if c >= 3:
                rgb_weights = channel_weights[:, :3, 0, 0].mean(dim=0)  # 平均batch的权重
                print(f"RGB通道注意力权重: R={rgb_weights[0]:.4f}, G={rgb_weights[1]:.4f}, B={rgb_weights[2]:.4f}")
                print(f"G通道权重占比: {rgb_weights[1] / (rgb_weights.sum() + 1e-8):.4f} (正常应≈0.33)")
            # 打印权重最大/最小的通道
            all_weights = channel_weights.mean(dim=0)[:, 0, 0]  # (C,)
            max_idx = all_weights.argmax().item()
            min_idx = all_weights.argmin().item()
            print(
                f"权重最大通道[{max_idx}]: {all_weights[max_idx]:.4f}, 权重最小通道[{min_idx}]: {all_weights[min_idx]:.4f}")
        self.debug_step += 1

        # 3. 加权每个通道：实现通道间的交互（重要通道加权高，次要通道加权低）
        return x * channel_weights


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        # 分组卷积（组内交互）
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=1, padding_mode="replicate")
        # 新增1x1点卷积（组间交互，关键修复）
        self.point_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0)  # 无分组，全通道交互
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.se = SEAttention(in_channels=dim)
        # 新增：调试step
        self.debug_step = 0

    def forward(self, x):
        input = x
        # ========== 新增：调试分组卷积前的特征 ==========
        if self.debug_step < 5:
            print_channel_distribution(x, f"Block输入 (dim={x.shape[1]})", self.debug_step)

        # 分组卷积
        x = self.conv(x)
        # ========== 新增：调试分组卷积后的特征 ==========
        if self.debug_step < 5:
            print_channel_distribution(x, f"分组卷积后 (groups=8)", self.debug_step)

        # 1x1点卷积
        x = self.point_conv(x)
        # ========== 新增：调试1x1卷积后的特征 ==========
        if self.debug_step < 5:
            print_channel_distribution(x, f"1x1卷积后 (组间融合)", self.debug_step)

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = self.se(x)
        x = input + self.drop_path(x)

        self.debug_step += 1
        return x


class LayerNorm(nn.Module):
    r""" 支持两种数据格式的LayerNorm：
    - channels_last：输入格式（N, H, W, C）
    - channels_first：输入格式（N, C, H, W）
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 缩放参数
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # 偏移参数
        self.eps = eps
        self.data_format = data_format
        assert self.data_format in ["channels_last", "channels_first"]
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            # 通道最后格式：直接用PyTorch原生LayerNorm
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # 通道优先格式：手动计算（因为PyTorch原生LayerNorm默认通道最后）
            u = x.mean(1, keepdim=True)  # 沿通道维度求均值（N, 1, H, W）
            s = (x - u).pow(2).mean(1, keepdim=True)  # 沿通道维度求方差
            x = (x - u) / torch.sqrt(s + self.eps)  # 归一化
            # 应用缩放和偏移（权重维度[C] → [C,1,1]，适配通道优先格式）
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x