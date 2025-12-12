import torch.nn as nn


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * (scale ** 2),
            kernel_size=3,
            padding=1,
            padding_mode="reflect"
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)

        self.post_smooth = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect"
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        x = self.post_smooth(x)
        return x


class FixedFinalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, source_img_channels=3):
        super().__init__()
        self.out_channels = out_channels
        self.inner_feat_dim = in_channels // 4

        # 1. 特征提取
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.inner_feat_dim * 2,
            kernel_size=3,
            padding=1,
            padding_mode="reflect"
        )
        self.relu = nn.ReLU(inplace=True)

        # 2. 上采样
        upsample_layers = []
        upsample_layers.append(UpsampleBlock(in_channels=self.inner_feat_dim * 2, scale=2))
        upsample_layers.append(UpsampleBlock(in_channels=self.inner_feat_dim * 2, scale=2))
        self.upsample = nn.Sequential(*upsample_layers)

        # 3. 重构层
        self.conv_output = nn.Conv2d(
            in_channels=self.inner_feat_dim * 2,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect"
        )

        # 其他层定义
        self.pos_enc1 = nn.Sequential(nn.Conv2d(source_img_channels, self.inner_feat_dim, 1), nn.ReLU())
        self.pos_enc2 = nn.Sequential(nn.Conv2d(source_img_channels, out_channels, 1), nn.ReLU())
        self.final_act = nn.Tanh()

    def forward(self, x, source_img):
        # 1. 特征提取
        x = self.conv1(x)
        x = self.relu(x)

        # 2. 上采样
        x = self.upsample(x)

        # 3. 重构层
        x = self.conv_output(x)

        # 4. 最终激活
        x = self.final_act(x)

        return x