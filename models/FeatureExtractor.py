import torch
import torch.nn as nn
import torch.nn.functional as F
from models.convBlock import SEAttention, Block


class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, drop_path=0.):
        super().__init__()
        # 优化通道扩展策略
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode="replicate")
        self.bn1 = nn.BatchNorm2d(out_channels)  # 添加BN

        # 替换MaxPool2d：定义第一个下采样卷积层（stride=2实现尺寸减半）
        self.downsample1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_channels)  # 保持BN风格一致
        )

        self.block1 = Block(dim=out_channels, drop_path=drop_path)

        # 渐进式通道扩展
        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, 3, padding=1, padding_mode="replicate")
        self.bn2 = nn.BatchNorm2d(out_channels * 2)

        # 替换MaxPool2d：定义第二个下采样卷积层（stride=2实现尺寸减半）
        self.downsample2 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels * 2, 3, stride=2, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(out_channels * 2)
        )

        self.block2 = Block(dim=out_channels * 2, drop_path=drop_path)

        # 最终通道调整
        self.conv3 = nn.Conv2d(out_channels * 2, out_channels * 2, 1, padding_mode="replicate")
        self.se_post = SEAttention(in_channels=out_channels * 2)

        # 移除原MaxPool2d层
        # self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 替换：用下采样卷积替代MaxPool2d
        x = self.downsample1(x)  # 尺寸减半，替代原self.pool(x)

        # 第二层
        x = self.block1(x)
        x = self.block1(x)

        # 第三层
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # 替换：用下采样卷积替代MaxPool2d
        x = self.downsample2(x)  # 尺寸减半，替代原self.pool(x)

        # 第四层
        x = self.block2(x)
        x = self.block2(x)
        x = self.conv3(x)
        x = self.se_post(x)

        return x

class FusionFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path=0.):
        super().__init__()

        # 分支1: 常规路径（移除 MaxPool2d，避免下采样）
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1, padding_mode="replicate"),  # stride=1 显式声明，确保无尺寸变化
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            Block(out_channels, drop_path)  # 移除 MaxPool2d，保留特征尺寸 32×32
        )

        # 分支2: 空洞卷积路径（扩大感受野，且尺寸仍为 32×32）
        self.dilated_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, stride=1, padding_mode="replicate"),  # 尺寸不变
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=4, dilation=4, stride=1, padding_mode="replicate"),  # 尺寸不变
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # 特征融合（通道拼接后仍保持 32×32）
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels * 2, 1, stride=1, padding_mode="replicate"),  # 1×1卷积仅融合通道，不改变尺寸
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),
            SEAttention(out_channels * 2)  # 注意力不改变尺寸
        )

    def forward(self, x):
        # 多尺度特征提取（双分支输出均为 32×32，无需自适应池化对齐）
        main_feat = self.main_branch(x)       # 输出: [B, out_channels, 32, 32]
        dilated_feat = self.dilated_branch(x) # 输出: [B, out_channels, 32, 32]

        # 特征融合（通道拼接，尺寸仍为 32×32）
        fused = torch.cat([main_feat, dilated_feat], dim=1)  # 输出: [B, 2*out_channels, 32, 32]
        return self.fusion(fused)                            # 最终输出: [B, 2*out_channels, 32, 32]