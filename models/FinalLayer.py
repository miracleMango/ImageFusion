
import torch.nn as nn


class FixedFinalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, source_img_channels=3):
        super().__init__()
        self.out_channels = out_channels  # 新增：记录输出通道数
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, padding_mode="replicate")
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.upsample1_conv = nn.Conv2d(in_channels // 2, 4 * (in_channels // 4), 3, padding=1,
                                        padding_mode="replicate")
        self.upsample1 = nn.PixelShuffle(2)
        self.upsample2_conv = nn.Conv2d(in_channels // 4, 4 * out_channels, 3, padding=1, padding_mode="replicate")
        self.upsample2 = nn.PixelShuffle(2)
        self.pos_enc1 = nn.Sequential(nn.Conv2d(source_img_channels, in_channels // 4, 1), nn.ReLU())
        self.pos_enc2 = nn.Sequential(nn.Conv2d(source_img_channels, out_channels, 1), nn.ReLU())
        self.final_act = nn.Tanh()

    def forward(self, x, source_img):
        # 调试：仅在输出通道数≥3时（VIS/融合解码器）执行RGB特征调试
        if self.out_channels >= 3 and x.shape[1] >= 3:
            print("===== 解码器输入特征的B通道对应特征 =====")
            x_b = x[:, 2, :, :]
            print(f"均值: {x_b.mean().item():.4f}, 范围: [{x_b.min().item():.4f}, {x_b.max().item():.4f}]")

        # 第一步：conv1
        x = self.conv1(x)
        x = self.relu(x)
        # 仅3通道解码器调试
        if self.out_channels >= 3 and x.shape[1] >= 3:
            print("===== 解码器conv1后B通道对应特征 =====")
            x_b = x[:, 2, :, :]
            print(f"均值: {x_b.mean().item():.4f}, 范围: [{x_b.min().item():.4f}, {x_b.max().item():.4f}]")

        # 第二步：第一次上采样
        x = self.upsample1_conv(x)
        x = self.relu(x)
        x = self.upsample1(x)
        # 仅3通道解码器调试
        if self.out_channels >= 3 and x.shape[1] >= 3:
            print("===== 解码器第一次上采样后B通道对应特征 =====")
            x_b = x[:, 2, :, :]
            print(f"均值: {x_b.mean().item():.4f}, 范围: [{x_b.min().item():.4f}, {x_b.max().item():.4f}]")

        # 第三步：第二次上采样
        x = self.upsample2_conv(x)
        x = self.tanh(x)
        x = self.upsample2(x)

        # 关键修改：仅3通道解码器（VIS/融合）调试B通道，IR解码器跳过
        if self.out_channels >= 3:
            print("===== 解码器第二次上采样后B通道 =====")
            x_b = x[:, 2, :, :]  # 3通道时取B通道（索引2）
            print(f"均值: {x_b.mean().item():.4f}, 范围: [{x_b.min().item():.4f}, {x_b.max().item():.4f}]")
        else:
            # IR解码器（单通道）仅打印整体分布，不按RGB拆分
            print("===== IR解码器第二次上采样后单通道 =====")
            print(f"均值: {x.mean().item():.4f}, 范围: [{x.min().item():.4f}, {x.max().item():.4f}]")

        # 最终激活
        x = self.final_act(x)

        # 最终输出调试：区分3通道/VIS和单通道/IR
        if self.out_channels >= 3:
            print("===== 解码器最终输出B通道 =====")
            x_b = x[:, 2, :, :]
            print(f"均值: {x_b.mean().item():.4f}, 范围: [{x_b.min().item():.4f}, {x_b.max().item():.4f}]")
        else:
            print("===== IR解码器最终输出单通道 =====")
            print(f"均值: {x.mean().item():.4f}, 范围: [{x.min().item():.4f}, {x.max().item():.4f}]")

        return x