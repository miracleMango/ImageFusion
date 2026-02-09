import torch
import torch.nn as nn
import math

class RotaryPositionEmbedding2D(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0, train_size: tuple = (256, 256)):
        super().__init__()
        assert head_dim % 4 == 0, f"二维RoPE的head_dim必须是4的倍数，当前输入为{head_dim}"
        self.head_dim = head_dim
        self.theta = theta
        self.freq_len = self.head_dim // 4
        # 新增：记录训练尺寸，用于外推时缩放坐标
        self.train_h, self.train_w = train_size

        # 新增：统一数据类型为float32，避免类型不匹配
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 4).float() / self.head_dim))
        self.register_buffer("inv_freq_x", inv_freq, persistent=False)
        self.register_buffer("inv_freq_y", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        batch_size, n_heads, seq_len, head_dim = x.shape
        assert head_dim == self.head_dim, f"输入head_dim{head_dim}与初始化{self.head_dim}不匹配"
        assert H * W == seq_len, f"二维尺寸H*W={H}*{W}={H*W}，与seq_len={seq_len}不相等"
        # 新增：校验H/W合法性
        assert H > 0 and W > 0, f"H/W必须为正整数，当前H={H}, W={W}"

        # 1. 生成坐标并缩放（核心：适配尺寸外推）
        y_coords = torch.arange(H, device=x.device, dtype=x.dtype)
        x_coords = torch.arange(W, device=x.device, dtype=x.dtype)
        # 按训练尺寸缩放坐标，回到训练分布（关键！）
        y_coords = y_coords * (self.train_h / H)  # 768→256: 767*(256/768)=255.666
        x_coords = x_coords * (self.train_w / W)  # 1024→256: 1023*(256/1024)=255.75
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()

        # 2. 计算角频率，新增数值裁剪（避免FP16溢出）
        freqs_x = torch.outer(x_flat, self.inv_freq_x.to(x.dtype))  # 替换einsum，更高效
        freqs_y = torch.outer(y_flat, self.inv_freq_y.to(x.dtype))
        # 裁剪角频率到合理范围（FP16安全范围：±1e4）
        freqs_x = torch.clamp(freqs_x, -1e4, 1e4)
        freqs_y = torch.clamp(freqs_y, -1e4, 1e4)

        # 3. 扩展维度，减少冗余unsqueeze
        freqs_x = freqs_x.unsqueeze(0).unsqueeze(0)  # [1,1,seq_len,freq_len]
        freqs_y = freqs_y.unsqueeze(0).unsqueeze(0)
        cos_x, sin_x = freqs_x.cos(), freqs_x.sin()
        cos_y, sin_y = freqs_y.cos(), freqs_y.sin()

        # 4. 维度拆分与旋转（逻辑不变，新增拼接后校验）
        x1, x2, y1, y2 = x[..., 0::4], x[..., 1::4], x[..., 2::4], x[..., 3::4]
        x1_rot = x1 * cos_x - x2 * sin_x
        x2_rot = x1 * sin_x + x2 * cos_x
        y1_rot = y1 * cos_y - y2 * sin_y
        y2_rot = y1 * sin_y + y2 * cos_y

        x_rotated = torch.cat([x1_rot, x2_rot, y1_rot, y2_rot], dim=-1)
        # 新增：校验拼接后维度
        assert x_rotated.shape[-1] == head_dim, f"拼接后维度{x_rotated.shape[-1]}≠原始{head_dim}"

        return x_rotated
