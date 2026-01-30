import torch.nn as nn
import math


class RotaryPositionEmbedding2D(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        """
        初始化二维RoPE位置编码层
        :param head_dim: 每个注意力头的特征维度，必须是4的倍数（二维RoPE硬约束）
        :param theta: 旋转基数，原论文推荐10000，控制频率衰减速度
        """
        super().__init__()
        # 二维RoPE核心约束：特征维度必须是4的倍数（分x/y各2份，形成奇偶对）
        assert head_dim % 4 == 0, f"二维RoPE的head_dim必须是4的倍数，当前输入为{head_dim}"
        self.head_dim = head_dim
        self.theta = theta
        # 计算单维度频率因子长度：head_dim / 4（x和y各占一份）
        self.freq_len = self.head_dim // 4

        # 预计算列坐标x的频率因子：1/(theta^(2i/head_dim))，i=0,1,...,freq_len-1
        inv_freq_x = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 4).float() / self.head_dim))
        # 预计算行坐标y的频率因子：与x独立，公式相同
        inv_freq_y = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 4).float() / self.head_dim))

        # 注册非持久化缓存：随模型迁移设备，不保存到state_dict，减少存储
        self.register_buffer("inv_freq_x", inv_freq_x, persistent=False)
        self.register_buffer("inv_freq_y", inv_freq_y, persistent=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        二维RoPE核心前向传播：生成二维网格坐标，执行分块旋转编码
        :param x: 输入张量，形状[batch_size, n_heads, seq_len, head_dim]
                  seq_len = H * W，为二维网格展平后的序列长度
        :param H: 二维网格的高度/行数，正整数
        :param W: 二维网格的宽度/列数，正整数
        :return: 二维旋转编码后的张量，形状与输入完全一致[batch_size, n_heads, H*W, head_dim]
        """
        # 1. 解包输入维度，执行严格的维度校验
        batch_size, n_heads, seq_len, head_dim = x.shape
        assert head_dim == self.head_dim, f"输入head_dim{head_dim}与初始化{self.head_dim}不匹配"
        assert H * W == seq_len, f"二维尺寸H*W={H}*{W}={H*W}，与seq_len={seq_len}不相等"

        # 2. 生成二维网格位置坐标 (x:列索引, y:行索引)
        # 生成行/列基础索引：y∈[0,H-1]，x∈[0,W-1]
        y_coords = torch.arange(H, device=x.device, dtype=self.inv_freq_x.dtype)
        x_coords = torch.arange(W, device=x.device, dtype=self.inv_freq_x.dtype)
        # 生成二维网格坐标：meshgrid输出(y_grid, x_grid)，形状均为[H, W]
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")  # indexing="ij"保证(i,j)对应(y,x)
        # 展平为序列形式：[H, W] → [H*W]，与seq_len维度对齐，共H*W个二维位置
        x_flat = x_grid.flatten()  # 列坐标展平，形状[seq_len]
        y_flat = y_grid.flatten()  # 行坐标展平，形状[seq_len]

        # 3. 计算行/列独立的角频率（广播相乘，避免重复计算）
        # 列x角频率：x_flat [seq_len] × inv_freq_x [freq_len] → [seq_len, freq_len]
        freqs_x = torch.einsum("i,j->ij", x_flat, self.inv_freq_x)
        # 行y角频率：y_flat [seq_len] × inv_freq_y [freq_len] → [seq_len, freq_len]
        freqs_y = torch.einsum("i,j->ij", y_flat, self.inv_freq_y)

        # 4. 扩展角频率维度，适配输入的广播机制（batch和n_heads维度）
        # 扩展为[1, 1, seq_len, freq_len]，可在batch_size/n_heads维度广播
        freqs_x = freqs_x.unsqueeze(0).unsqueeze(0)
        freqs_y = freqs_y.unsqueeze(0).unsqueeze(0)
        # 计算余弦/正弦值，为旋转做准备
        cos_x, sin_x = freqs_x.cos(), freqs_x.sin()
        cos_y, sin_y = freqs_y.cos(), freqs_y.sin()

        # 5. 核心：特征维度分块，拆分为x/y对应的奇偶维度对
        # head_dim=4*freq_len，分4块：x1(0::4), x2(1::4), y1(2::4), y2(3::4)
        x1 = x[..., 0::4]  # 列x旋转-偶数维度，形状[B, n_heads, seq_len, freq_len]
        x2 = x[..., 1::4]  # 列x旋转-奇数维度，形状与x1一致
        y1 = x[..., 2::4]  # 行y旋转-偶数维度，形状与x1一致
        y2 = x[..., 3::4]  # 行y旋转-奇数维度，形状与x1一致

        # 6. 二维旋转操作：行、列坐标分别执行旋转（与一维RoPE公式一致）
        # 列x对应的旋转
        x1_rot = x1 * cos_x - x2 * sin_x
        x2_rot = x1 * sin_x + x2 * cos_x
        # 行y对应的旋转
        y1_rot = y1 * cos_y - y2 * sin_y
        y2_rot = y1 * sin_y + y2 * cos_y

        # 7. 拼接旋转后的分块，恢复原特征维度head_dim
        # 按最后一维拼接：[x1_rot, x2_rot, y1_rot, y2_rot] → 恢复4*freq_len=head_dim
        x_rotated = torch.cat([x1_rot, x2_rot, y1_rot, y2_rot], dim=-1)

        return x_rotated
