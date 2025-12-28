import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.contrib.models.common import DropPath


class EnhancedAttention(nn.Module):
    """增强版注意力模块：全局池化 + 局部小核卷积 + 轴向稀疏自注意力 + 残差分支（四分支融合）"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        mid_channels = in_channels // reduction

        # ========== 分支1：全局池化分支 ==========
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, in_channels),
            nn.Sigmoid()
        )

        # ========== 分支2：两层局部小核卷积分支 ==========
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0), groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

        # ========== 分支3：轴向稀疏自注意力分支 ==========
        self.axial_head_dim = mid_channels
        self.row_proj = nn.Linear(in_channels, 3 * self.axial_head_dim)
        self.col_proj = nn.Linear(self.axial_head_dim, 3 * self.axial_head_dim)
        self.axial_out = nn.Sequential(
            nn.Linear(self.axial_head_dim, in_channels),
            nn.Sigmoid()
        )
        self.register_buffer('axial_scale', torch.sqrt(torch.tensor(self.axial_head_dim, dtype=torch.float32)))

        # ========== 分支4：残差分支 ==========
        # 残差分支无需额外层，仅需为其分配融合权重
        # ========== 四分支融合权重 ==========
        self.fusion_weight = nn.Parameter(torch.ones(4) / 4)  # 初始均等权重：[1/4, 1/4, 1/4, 1/4]
        self.softmax = nn.Softmax(dim=0)

    def axial_self_attention(self, x):
        """轴向稀疏自注意力"""
        b, c, h, w = x.shape
        
        # 1. 行注意力（沿宽度维度）
        x_row = x.permute(0, 2, 1, 3).reshape(b * h, c, w)
        x_row = x_row.permute(0, 2, 1)
        q_r, k_r, v_r = self.row_proj(x_row).chunk(3, dim=-1)
        attn_row = (q_r @ k_r.transpose(-2, -1)) / self.axial_scale
        attn_row = F.softmax(attn_row, dim=-1)
        out_row = (attn_row @ v_r).reshape(b, h, w, self.axial_head_dim)
        
        # 2. 列注意力（沿高度维度）
        x_col = out_row.permute(0, 2, 1, 3).reshape(b * w, h, self.axial_head_dim)
        q_c, k_c, v_c = self.col_proj(x_col).chunk(3, dim=-1)
        attn_col = (q_c @ k_c.transpose(-2, -1)) / self.axial_scale
        attn_col = F.softmax(attn_col, dim=-1)
        out_col = (attn_col @ v_c).reshape(b, w, h, self.axial_head_dim).permute(0, 2, 1, 3)
        
        # 3. 融合生成通道权重
        axial_out = self.axial_out(out_col)
        axial_weight = axial_out.mean(dim=(1, 2)).reshape(b, c, 1, 1)
        return axial_weight

    def forward(self, x):
        b, c, h, w = x.shape
        # 分支1：全局池化权重
        global_stats = self.gap(x).view(b, c)
        global_weight = self.global_fc(global_stats).view(b, c, 1, 1)

        # 分支2：局部卷积权重
        local_weight = self.local_conv(x)

        # 分支3：轴向自注意力权重
        axial_weight = self.axial_self_attention(x)

        # 分支4：残差分支权重
        # 残差分支权重为全1（代表原始特征无衰减），形状与其他分支一致：(B, C, 1, 1)
        residual_weight = torch.ones(b, c, 1, 1).to(x.device)

        # 四分支融合
        fusion_weights = self.softmax(self.fusion_weight)  # 归一化4个权重，和为1
        final_weight = (
            fusion_weights[0] * global_weight +
            fusion_weights[1] * local_weight +
            fusion_weights[2] * axial_weight +
            fusion_weights[3] * residual_weight  # 新增残差分支的加权
        )

        # 注意力加权：原始特征 × 融合后的权重
        return x * final_weight

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=1, padding_mode="replicate")
        self.point_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.se = EnhancedAttention(in_channels=dim)

    def forward(self, x):
        input = x
        x = self.conv(x)
        x = self.point_conv(x)

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

        return x


class LayerNorm(nn.Module):
    r""" 支持两种数据格式的LayerNorm """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert self.data_format in ["channels_last", "channels_first"]
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
