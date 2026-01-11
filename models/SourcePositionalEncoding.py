import torch
import torch.nn as nn
import torch.nn.functional as F

class SourcePositionalEncoding2D(nn.Module):
    """基于源图像下采样的2D RoPE旋转位置编码（适配动态输入通道）"""
    def __init__(self, in_channels, channels, max_size=256):
        super().__init__()
        self.channels = channels  # 目标通道数（需为偶数，RoPE分组要求）
        self.max_size = max_size

        # 校验通道数：确保为偶数，满足RoPE两两分组旋转的要求
        if self.channels % 2 != 0:
            raise ValueError(f"channels must be even for RoPE, got {self.channels}")

        # 投影层：将动态输入通道（in_channels）适配到目标通道数（channels）
        self.projection = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)

        # 预计算频率因子（RoPE核心，对应10000^(-2i/d)，扩展到2D）
        self.register_buffer(
            "freqs",
            1.0 / torch.pow(10000.0, 2 * torch.arange(0, self.channels // 2) / self.channels),
            persistent=False
        )

    def _get_2d_rotary_pos_emb(self, h, w, device):
        """生成2D位置的旋转因子（融合高度H和宽度W维度，修正维度对齐）"""
        # 1. 生成2D位置索引（行索引：h_idx，列索引：w_idx）
        h_idx = torch.arange(h, device=device).unsqueeze(-1)  # [H, 1] → 修正：用unsqueeze(-1)确保维度末尾扩展
        w_idx = torch.arange(w, device=device).unsqueeze(-1)  # [W, 1] → 修正：统一维度扩展方式

        # 2. 计算每个位置的角度值（融合H、W两个维度的位置信息）
        # 修正：self.freqs 形状 [C//2] → 转为 [1, C//2]，避免与h_idx/w_idx的维度1冲突
        freqs_expanded = self.freqs.unsqueeze(0)  # [1, C//2]

        # 修正：h_theta 形状 [H, C//2]，w_theta 形状 [W, C//2] → 广播对齐正确
        h_theta = h_idx * freqs_expanded  # [H, 1] * [1, C//2] → [H, C//2]
        w_theta = w_idx * freqs_expanded  # [W, 1] * [1, C//2] → [W, C//2]

        # 修正：扩展为2D空间形状 [H, W, C//2] → 融合H和W的位置信息
        h_theta = h_theta.unsqueeze(1).expand(-1, w, -1)  # [H, 1, C//2] → [H, W, C//2]
        w_theta = w_theta.unsqueeze(0).expand(h, -1, -1)  # [1, W, C//2] → [H, W, C//2]
        theta = h_theta + w_theta  # [H, W, C//2] → 融合2D位置的总角度

        # 3. 计算余弦和正弦值（旋转矩阵的核心元素）
        cos_theta = torch.cos(theta)  # [H, W, C//2]
        sin_theta = torch.sin(theta)  # [H, W, C//2]

        # 4. 重塑为适配特征图的形状 [1, C, H, W]（便于后续广播运算）
        # 通道维度重组：[C//2, 2] → [C]，对应两两分组的旋转矩阵
        cos_theta = cos_theta.permute(2, 0, 1).unsqueeze(0)  # [1, C//2, H, W]
        sin_theta = sin_theta.permute(2, 0, 1).unsqueeze(0)  # [1, C//2, H, W]
        cos_theta = torch.repeat_interleave(cos_theta, 2, dim=1)  # [1, C, H, W]
        sin_theta = torch.repeat_interleave(sin_theta, 2, dim=1)  # [1, C, H, W]

        return cos_theta, sin_theta

    def _apply_2d_rope(self, x, cos_theta, sin_theta):
        """对2D特征图执行RoPE旋转操作（核心：分组建旋转，保持正交性）"""
        # 1. 通道分组：将特征图通道分为偶数组和奇数组（两两一组）
        x_even = x[:, ::2, :, :]  # 偶数索引通道（0,2,4...）→ [B, C//2, H, W]
        x_odd = x[:, 1::2, :, :]  # 奇数索引通道（1,3,5...）→ [B, C//2, H, W]

        # 2. 执行旋转操作（对应2D旋转矩阵的乘法，等价于复数旋转）
        # 旋转后偶数通道：x_even*cos - x_odd*sin
        # 旋转后奇数通道：x_even*sin + x_odd*cos
        x_rot_even = x_even * cos_theta[:, ::2, :, :] - x_odd * sin_theta[:, ::2, :, :]
        x_rot_odd = x_even * sin_theta[:, ::2, :, :] + x_odd * cos_theta[:, ::2, :, :]

        # 3. 拼接旋转后的通道，恢复原特征图形状 [B, C, H, W]
        x_rot = torch.stack([x_rot_even, x_rot_odd], dim=2)  # [B, C//2, 2, H, W]
        x_rot = x_rot.permute(0, 2, 1, 3, 4).reshape(x.shape)  # [B, C, H, W]

        return x_rot

    def forward(self, x, source_img):
        """
        Args:
            x: 特征图 [B, C, H, W]（C需与self.channels一致）
            source_img: 源图像 [B, in_channels, H_source, W_source]
        Returns:
            x + pos_encoding: 融入2D RoPE位置信息的特征图
        """
        b, c, h, w = x.shape
        device = x.device

        # 校验特征图通道数与目标通道数一致
        if c != self.channels:
            raise ValueError(f"feature channel {c} must match self.channels {self.channels}")

        # 步骤1：源图像预处理（下采样+通道投影）
        # 1.1 下采样源图像到特征图尺寸（保持空间对齐）
        source_down = F.interpolate(
            source_img,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )
        # 1.2 投影到目标通道数（适配动态输入通道）
        source_proj = self.projection(source_down)  # [B, C, H, W]

        # 步骤2：生成2D RoPE旋转因子（余弦+正弦）
        cos_theta, sin_theta = self._get_2d_rotary_pos_emb(h, w, device)

        # 步骤3：对预处理后的源图像特征执行2D RoPE编码（得到pos_encoding）
        pos_encoding = self._apply_2d_rope(source_proj, cos_theta, sin_theta)

        # 步骤4：返回特征图与RoPE位置编码的加和（保持原输出形式）
        return x + pos_encoding
