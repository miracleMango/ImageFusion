import torch
import torch.nn as nn
import torch.nn.functional as F
from models.convBlock import EnhancedAttention, Block

class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, drop_path=0.):
        super().__init__()
        # 优化通道扩展策略
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode="replicate")
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 第1个下采样阶段（stage1）：通道=out_channels，尺寸减半
        self.downsample1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_channels)
        )
        # 加深：stage1堆叠4个Block（原2个）
        self.stage1 = nn.Sequential(*[Block(dim=out_channels, drop_path=drop_path) for _ in range(4)])

        # 第2个下采样阶段（stage2）：通道=2*out_channels，尺寸再减半
        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, 3, padding=1, padding_mode="replicate")
        self.bn2 = nn.BatchNorm2d(out_channels * 2)
        self.downsample2 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels * 2, 3, stride=2, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_channels * 2)
        )
        # 加深：stage2堆叠6个Block（原2个）
        self.stage2 = nn.Sequential(*[Block(dim=out_channels * 2, drop_path=drop_path) for _ in range(6)])

        # 最终通道调整+SE
        self.conv_final = nn.Conv2d(out_channels * 2, out_channels * 2, 1, padding_mode="replicate")
        self.se_post = EnhancedAttention(in_channels=out_channels * 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # stage1：下采样+4个Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.downsample1(x)
        x = self.stage1(x)

        # stage2：通道扩展+下采样+6个Block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.downsample2(x)
        x = self.stage2(x)

        # 最终调整
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
