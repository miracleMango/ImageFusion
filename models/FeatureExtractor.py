import torch
import torch.nn as nn
import torch.nn.functional as F
from models.convBlock import EnhancedAttention, Block

class CNNFeatureExtractorWithGlobal(nn.Module):
    """改造后的Patch特征提取器：融入全局+位置特征，不修改原有核心逻辑"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, drop_path=0.):
        super().__init__()
        # 完全复用原有初始化逻辑（不修改核心代码）
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode="replicate")
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.downsample1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_channels)
        )
        self.stage1 = nn.Sequential(*[Block(dim=out_channels, drop_path=drop_path) for _ in range(4)])

        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, 3, padding=1, padding_mode="replicate")
        self.bn2 = nn.BatchNorm2d(out_channels * 2)
        self.downsample2 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels * 2, 3, stride=2, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_channels * 2)
        )
        self.stage2 = nn.Sequential(*[Block(dim=out_channels * 2, drop_path=drop_path) for _ in range(6)])

        self.conv_final = nn.Conv2d(out_channels * 2, out_channels * 2, 1, padding_mode="replicate")
        self.se_post = EnhancedAttention(in_channels=out_channels * 2)
        self.relu = nn.ReLU(inplace=False)
        
        # 新增：全局特征融合层（匹配两个关键节点的特征维度）
        # 节点1：stage1输出后（维度=out_channels）
        self.fusion_stage1 = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),  # 全局特征维度→stage1输出维度
            nn.LayerNorm(out_channels)
        )
        # 节点2：stage2输出后（维度=out_channels*2）
        self.fusion_stage2 = nn.LayerNorm(out_channels * 2)

    def _broadcast_global_feat(self, global_feat, patch_feat_shape):
        """
        全局特征向量→空间广播为patch特征图形状，适配融合
        input：global_feat [B, C]，patch_feat_shape [B, C_patch, Hp, Wp]
        output：broadcast_feat [B, C_patch, Hp, Wp]
        """
        B, C_patch, Hp, Wp = patch_feat_shape
        global_feat = global_feat.view(B, -1, 1, 1)  # [B, C, 1, 1]
        broadcast_feat = global_feat.expand(B, C_patch, Hp, Wp)  # 空间广播
        return broadcast_feat

    def forward(self, x, global_pos_feat):
        """
        输入：
            x - patch图像（ir/vis模态，[B, in_channels, Hp, Wp]）
            global_pos_feat - 融入位置信息的全局特征（[B, out_channels*2]）
        输出：
            x - 融入全局信息的patch特征（[B, out_channels*2, Hp//4, Wp//4]）
        """
        # ---------------------- 原有逻辑：stage1  ----------------------
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.downsample1(x)
        x = self.stage1(x)  # stage1输出：[B, out_channels, Hp//2, Wp//2]
        
        # ---------------------- 新增：第一个融合节点（stage1后） ----------------------
        # 1. 全局特征映射到stage1输出维度
        global_feat_stage1 = self.fusion_stage1(global_pos_feat)
        # 2. 空间广播适配patch特征形状
        broadcast_feat_stage1 = self._broadcast_global_feat(global_feat_stage1, x.shape)
        # 3. 特征融合（逐元素相加，残差连接，不破坏原有patch特征）
        x = x + self.relu(broadcast_feat_stage1)
        
        # ---------------------- 原有逻辑：stage2  ----------------------
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.downsample2(x)
        x = self.stage2(x)  # stage2输出：[B, out_channels*2, Hp//4, Wp//4]
        
        # ---------------------- 新增：第二个融合节点（stage2后） ----------------------
        # 1. 全局特征层归一化（匹配stage2输出维度）
        global_feat_stage2 = self.fusion_stage2(global_pos_feat)
        # 2. 空间广播适配patch特征形状
        broadcast_feat_stage2 = self._broadcast_global_feat(global_feat_stage2, x.shape)
        # 3. 特征融合（逐元素相加，残差连接）
        x = x + self.relu(broadcast_feat_stage2)
        
        # ---------------------- 原有逻辑：最终调整  ----------------------
        x = self.conv_final(x)
        x = self.se_post(x)
        
        return x

class FusionFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path=0.):
        super().__init__()

        # 分支1: 常规路径（堆叠4个Block，尺寸/通道不变）
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, 3, padding=1, stride=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),
            # 加深：从1个Block→4个（维度不变）
            *[Block(dim=out_channels * 2, drop_path=drop_path) for _ in range(4)]
        )

        # 分支2: 空洞卷积路径（新增Block堆叠，尺寸/通道不变）
        self.dilated_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, 3, padding=2, dilation=2, stride=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),
            nn.Conv2d(out_channels * 2, out_channels * 2, 3, padding=4, dilation=4, stride=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),
            # 加深：新增3个Block（维度不变）
            *[Block(dim=out_channels * 2, drop_path=drop_path) for _ in range(3)]
        )

        # 特征融合（完全复用你原有的结构，维度不变）
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels * 2, 1, stride=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),
            # 融合后新增1个Block（加深，维度不变）
            Block(out_channels * 2, drop_path),
            EnhancedAttention(out_channels * 2)
        )

    def forward(self, x):
        main_feat = self.main_branch(x)
        dilated_feat = self.dilated_branch(x)

        fused = torch.cat([main_feat, dilated_feat], dim=1)
        out = self.fusion(fused)
        return out
