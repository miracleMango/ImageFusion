import torch
import torch.nn as nn
import torch.nn.functional as F
from models.FeatureExtractor import CNNFeatureExtractor, FusionFeatureExtractor
from models.FinalLayer import FixedFinalLayer
from models.crossAttn import MultiHeadCrossAttentionWithSourcePE, MultiHeadCrossAttentionFusionWithSourcePE
from models.debug import print_channel_distribution
from models.loss import Loss


class ImageFusionNetworkWithSourcePE(nn.Module):
    def __init__(self, vis_img_channels=3, ir_img_channels=1, feature_channels=64, num_heads=8,
                 use_position_encoding=True):
        super().__init__()
        # 1. 特征提取器
        self.ir_extractor = CNNFeatureExtractor(ir_img_channels, feature_channels)
        self.vis_extractor = CNNFeatureExtractor(vis_img_channels, feature_channels)
        self.fusion_extractor = FusionFeatureExtractor(feature_channels * 4, feature_channels)

        # 2. 交叉注意力模块（显式传入源图像通道参数）
        self.cross_attn_vis = MultiHeadCrossAttentionWithSourcePE(
            feature_channels, num_heads, use_position_encoding=use_position_encoding,
            src_q_channels=vis_img_channels, src_k_channels=vis_img_channels, src_v_channels=vis_img_channels
            # vis源图像3通道
        )
        self.cross_attn_ir = MultiHeadCrossAttentionWithSourcePE(
            feature_channels, num_heads, use_position_encoding=use_position_encoding,
            src_q_channels=ir_img_channels, src_k_channels=ir_img_channels, src_v_channels=ir_img_channels
            # ir源图像1通道
        )

        # 3. 融合分支的交叉注意力模块（传入fusion分支的源图像通道）
        self.cross_attn_fusion = MultiHeadCrossAttentionFusionWithSourcePE(
            feature_channels, num_heads, use_position_encoding=use_position_encoding,
            src_qk_channels=ir_img_channels + vis_img_channels,  # qk是ir+vis拼接（1+3=4通道）
            src_v_channels=vis_img_channels  # v是ir转3通道+vis平均（3通道）
        )

        # 4. Final Layer（原逻辑保留）
        self.final_vis = FixedFinalLayer(
            in_channels=feature_channels * 2,
            out_channels=vis_img_channels,
            source_img_channels=vis_img_channels
        )
        self.final_ir = FixedFinalLayer(
            in_channels=feature_channels * 2,
            out_channels=ir_img_channels,
            source_img_channels=ir_img_channels
        )
        self.final_fusion = FixedFinalLayer(
            in_channels=feature_channels * 2,
            out_channels=vis_img_channels,
            source_img_channels=vis_img_channels
        )

        # 5. 损失函数
        self.compute_loss = Loss(device="cuda" if torch.cuda.is_available() else "cpu")

        # 调试计数器
        self.debug_step = 0
        self.max_debug_steps = 5  # 只调试前5个step

    def _debug_print(self, tensor, name, step):
        """调试打印函数"""
        if self.debug_step < self.max_debug_steps:
            print(f"Step {step}: {name} - 范围: [{tensor.min().item():.3f}, {tensor.max().item():.3f}], "
                  f"均值: {tensor.mean().item():.3f}, 形状: {tensor.shape}")

    def _enhance_ir_features(self, fir, fvis):
        """增强红外特征 - 强调热辐射信息"""
        # 使用注意力机制突出红外独特特征
        b, c, h, w = fir.shape
        ir_attention = torch.sigmoid(fir.mean(dim=1, keepdim=True))  # 基于强度的注意力
        enhanced_ir = fir * (1 + ir_attention)  # 增强显著区域

        # 调试信息
        if self.debug_step < self.max_debug_steps:
            print(
                f"Step {self.debug_step}: IR特征增强 - 注意力范围: [{ir_attention.min().item():.3f}, {ir_attention.max().item():.3f}]")
            print(
                f"Step {self.debug_step}: IR特征增强 - 输入范围: [{fir.min().item():.3f}, {fir.max().item():.3f}]")
            print(
                f"Step {self.debug_step}: IR特征增强 - 输出范围: [{enhanced_ir.min().item():.3f}, {enhanced_ir.max().item():.3f}]")

        return enhanced_ir

    def _enhance_vis_features(self, fvis, fir):
        """增强可见光特征 - 强调纹理细节"""
        # 计算可见光与红外的差异，突出细节信息
        detail_attention = torch.sigmoid(torch.abs(fvis - fir).mean(dim=1, keepdim=True))
        enhanced_vis = fvis * (1 + detail_attention)  # 增强细节区域

        # 调试信息
        if self.debug_step < self.max_debug_steps:
            print(
                f"Step {self.debug_step}: VIS特征增强 - 注意力范围: [{detail_attention.min().item():.3f}, {detail_attention.max().item():.3f}]")
            print(
                f"Step {self.debug_step}: VIS特征增强 - 输入范围: [{fvis.min().item():.3f}, {fvis.max().item():.3f}]")
            print(
                f"Step {self.debug_step}: VIS特征增强 - 输出范围: [{enhanced_vis.min().item():.3f}, {enhanced_vis.max().item():.3f}]")

        return enhanced_vis

    def forward(self, img_ir, img_vis):
        # 调试信息
        if self.debug_step < self.max_debug_steps:
            print(f"\n=== 前向传播调试 Step {self.debug_step} ===")
            # ========== 新增：调试输入RGB分布 ==========
            print_channel_distribution(img_vis, "输入可见光(RGB)", self.debug_step, is_rgb=True)
            print_channel_distribution(img_ir, "输入红外(单通道)", self.debug_step)

        # 保存原始输入用于位置编码
        source_ir = img_ir  # [B,1,H,W]
        source_vis = img_vis  # [B,3,H,W]

        # 提取特征（原逻辑保留）
        fir = self.ir_extractor(img_ir)
        fvis = self.vis_extractor(img_vis)

        self._debug_print(fir, "IR特征提取器输出", self.debug_step)
        self._debug_print(fvis, "VIS特征提取器输出", self.debug_step)
        # ========== 新增：调试特征提取后的RGB相关分布 ==========
        if self.debug_step < self.max_debug_steps:
            print_channel_distribution(fvis, "VIS特征提取后", self.debug_step)  # 特征通道前3模拟RGB

        # 特征增强（原逻辑保留）
        fir_enhanced = self._enhance_ir_features(fir, fvis)
        fvis_enhanced = self._enhance_vis_features(fvis, fir)

        fvis_concat = torch.cat([fir_enhanced, fvis_enhanced], dim=1)
        ffusion = self.fusion_extractor(fvis_concat)

        self._debug_print(fvis_concat, "特征拼接", self.debug_step)
        self._debug_print(ffusion, "融合特征提取器输出", self.debug_step)

        # 获取特征图尺寸用于源图像下采样
        _, _, h_feat, w_feat = fir.shape

        # 下采样源图像到特征图尺寸
        source_ir_down = F.interpolate(source_ir, size=(h_feat, w_feat), mode='bilinear',
                                       align_corners=False)  # [B,1,h,w]
        source_vis_down = F.interpolate(source_vis, size=(h_feat, w_feat), mode='bilinear',
                                        align_corners=False)  # [B,3,h,w]

        # 融合分支qk的源图像：拼接（1+3=4通道）
        source_fusion_qk = torch.cat([source_ir_down, source_vis_down], dim=1)  # [B,4,h,w]

        # 交叉注意力交互（原逻辑保留）
        attn_vis_out, _ = self.cross_attn_vis(
            fvis, ffusion, ffusion,
            source_vis_down, source_vis_down, source_vis_down
        )
        self._debug_print(attn_vis_out, "VIS注意力输出", self.debug_step)

        img_vis_pred = self.final_vis(attn_vis_out, source_img=img_vis)
        self._debug_print(img_vis_pred, "最终VIS重建", self.debug_step)
        # ========== 新增：调试重建VIS的RGB分布 ==========
        if self.debug_step < self.max_debug_steps:
            print_channel_distribution(img_vis_pred, "重建可见光(RGB)", self.debug_step, is_rgb=True)

        attn_ir_out, _ = self.cross_attn_ir(
            fir, ffusion, ffusion,
            source_ir_down, source_ir_down, source_ir_down
        )
        self._debug_print(attn_ir_out, "IR注意力输出", self.debug_step)

        img_ir_pred = self.final_ir(attn_ir_out, source_img=img_ir)
        self._debug_print(img_ir_pred, "最终IR重建", self.debug_step)

        # 融合分支：拼接fir和fvis作为q和k
        fir_fvis_concat = torch.cat([fir, fvis], dim=1)

        # 融合分支v的源图像：红外1通道转3通道后和可见光平均（核心修改）
        source_ir_down_3ch = torch.cat([source_ir_down, source_ir_down, source_ir_down], dim=1)  # 1→3通道
        source_fusion_v = (source_ir_down_3ch + source_vis_down) / 2  # [B,3,h,w]
        # ========== 新增：调试融合源图像的RGB分布 ==========
        if self.debug_step < self.max_debug_steps:
            print_channel_distribution(source_fusion_v, "融合源图像(RGB)", self.debug_step, is_rgb=True)

        attn_fusion_out, _ = self.cross_attn_fusion(
            fir_fvis_concat, fir_fvis_concat, ffusion,
            source_fusion_qk, source_fusion_qk, source_fusion_v
        )
        self._debug_print(attn_fusion_out, "融合注意力输出", self.debug_step)

        # 融合源图像：红外转3通道后和可见光平均
        source_fusion = (source_ir_down_3ch + source_vis_down) / 2
        img_fusion_pred = self.final_fusion(attn_fusion_out, source_img=source_fusion)
        self._debug_print(img_fusion_pred, "最终融合图像", self.debug_step)
        # ========== 新增：调试最终融合图像的RGB分布 ==========
        if self.debug_step < self.max_debug_steps:
            print_channel_distribution(img_fusion_pred, "最终融合图像(RGB)", self.debug_step, is_rgb=True)

        # 最终输出总结（原逻辑保留，新增RGB比值）
        if self.debug_step < self.max_debug_steps:
            print(f"\n=== 最终输出总结 Step {self.debug_step} ===")
            print(
                f"重建IR范围: [{img_ir_pred.min().item():.3f}, {img_ir_pred.max().item():.3f}], 均值: {img_ir_pred.mean().item():.3f}")
            print(
                f"重建VIS范围: [{img_vis_pred.min().item():.3f}, {img_vis_pred.max().item():.3f}], 均值: {img_vis_pred.mean().item():.3f}")
            print(
                f"融合图像范围: [{img_fusion_pred.min().item():.3f}, {img_fusion_pred.max().item():.3f}], 均值: {img_fusion_pred.mean().item():.3f}")

            # ========== 新增：计算RGB通道失衡指标 ==========
            if img_fusion_pred.shape[1] == 3:
                fusion_r = img_fusion_pred[:, 0, :, :].mean().item()
                fusion_g = img_fusion_pred[:, 1, :, :].mean().item()
                fusion_b = img_fusion_pred[:, 2, :, :].mean().item()
                print(f"融合图像RGB均值: R={fusion_r:.4f}, G={fusion_g:.4f}, B={fusion_b:.4f}")
                print(f"R/G比值: {fusion_r / (fusion_g + 1e-8):.4f}, B/G比值: {fusion_b / (fusion_g + 1e-8):.4f}")
                if fusion_g > fusion_r * 1.5 or fusion_g > fusion_b * 1.5:
                    print("❌ G通道过度激活，存在偏绿问题！")
                else:
                    print("✅ RGB通道分布基本均衡")

            # 检查是否使用了Tanh激活函数
            if img_ir_pred.min() >= -1.0 and img_ir_pred.max() <= 1.0:
                print("✅ Tanh激活函数正常工作")
            else:
                print("❌ Tanh激活函数异常，输出范围超出[-1,1]")

            # 检查输出范围是否过小
            ir_range = img_ir_pred.max() - img_ir_pred.min()
            vis_range = img_vis_pred.max() - img_vis_pred.min()
            fusion_range = img_fusion_pred.max() - img_fusion_pred.min()

            if ir_range < 0.1 or vis_range < 0.1 or fusion_range < 0.1:
                print("⚠️  警告：输出范围过小，可能存在压缩问题")

            print("=" * 80 + "\n")

        self.debug_step += 1

        return {
            "img_vis_pred": img_vis_pred,
            "img_ir_pred": img_ir_pred,
            "img_fusion_pred": img_fusion_pred,
            "features": {"fir": fir, "fvis": fvis, "ffusion": ffusion}
        }

