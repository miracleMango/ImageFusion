import torch.nn as nn

from models.convBlock import SEAttention, Block


class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, drop_path=0.):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode="replicate")
        self.relu = nn.ReLU()
        self.block = Block(dim=out_channels, drop_path=drop_path)  # 用新Block（传统卷积）
        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, 1, 1, 0)
        self.pool = nn.MaxPool2d(2, 2)
        self.se_post = SEAttention(in_channels=out_channels * 2)  # 保留全局通道注意力

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.block(x)
        x = self.block(x)
        x = self.conv2(x)
        x = self.se_post(x)
        x = self.pool(x)
        return x


class FusionFeatureExtractor(nn.Module):
    def __init__(self, concat_channels, out_channels, drop_path=0.):
        super().__init__()
        self.conv1 = nn.Conv2d(concat_channels, out_channels * 8, 1, 1, 0)
        self.relu = nn.ReLU()
        self.block = Block(dim=out_channels * 8, drop_path=drop_path)  # 用新Block（传统卷积）
        self.pool = nn.MaxPool2d(2, 2)
        self.channel_linear = nn.Conv2d(out_channels * 8, out_channels * 2, 1, 1, 0)
        self.se_final = SEAttention(in_channels=out_channels * 2)  # 保留全局通道注意力

    def forward(self, fvis_concat):
        x = self.conv1(fvis_concat)
        x = self.relu(x)
        x = self.block(x)
        x = self.block(x)
        x = self.channel_linear(x)
        x = self.se_final(x)
        return x
