import torch.nn as nn
import torch.nn.functional as F

class SourcePositionalEncoding2D(nn.Module):
    """基于源图像下采样的2D位置编码（适配动态输入通道）"""
    def __init__(self, in_channels, channels, max_size=256):  # 核心：in_channels改为参数
        super().__init__()
        self.channels = channels
        self.max_size = max_size

        # 投影层适配输入通道（原固定3→现在动态in_channels）
        self.projection = nn.Conv2d(in_channels, channels, kernel_size=1)

    def forward(self, x, source_img):
        """
        Args:
            x: 特征图 [B, C, H, W]
            source_img: 源图像 [B, in_channels, H_source, W_source]
        """
        b, c, h, w = x.shape

        # 将源图像下采样到特征图尺寸
        source_down = F.interpolate(source_img, size=(h, w), mode='bilinear', align_corners=False)

        # 投影到目标通道数（输入通道已适配）
        pos_encoding = self.projection(source_down)

        return x + pos_encoding
