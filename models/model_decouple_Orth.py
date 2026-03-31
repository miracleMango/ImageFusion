import torch
import torch.nn as nn
import torch.nn.functional as F
from models.FeatureExtractor import CNNFeatureExtractor, FusionFeatureExtractor
from models.FinalLayer import FixedFinalLayer
from models.crossAttn import MultiHeadCrossAttentionWithSourcePE, MultiHeadSelfAttentionWithSourcePE
from models.HeatSourceMaskGenerator import HeatSourceMaskGenerator
from models.loss import Loss

class ImageFusionNetworkWithDecoupleOrth(nn.Module):
    def __init__(self, vis_img_channels=3, ir_img_channels=1, feature_channels=64, num_heads=8,
                 use_position_encoding=True):
        super().__init__()
        self.feature_channels = feature_channels
        self.num_heads = num_heads
        self.use_position_encoding = use_position_encoding
        self.attn_cycles = 3  # 注意力循环轮数，与模块组数一致

        # ======================== 原有模块1：特征提取器（完全保持不变）========================
        self.ir_extractor = CNNFeatureExtractor(ir_img_channels, feature_channels)
        self.vis_extractor = CNNFeatureExtractor(vis_img_channels, feature_channels)
        self.fusion_extractor = FusionFeatureExtractor(feature_channels * 4, feature_channels)

        # ======================== 原有模块2：3组独立Self-Attention（完全保持不变）========================
        self.self_attn_vis_cycles = nn.ModuleList()
        for _ in range(self.attn_cycles):
            self.self_attn_vis_cycles.append(MultiHeadSelfAttentionWithSourcePE(feature_channels, num_heads, use_position_encoding=use_position_encoding, num_blocks=1))
        self.self_attn_ir_cycles = nn.ModuleList()
        for _ in range(self.attn_cycles):
            self.self_attn_ir_cycles.append(MultiHeadSelfAttentionWithSourcePE(feature_channels, num_heads, use_position_encoding=use_position_encoding, num_blocks=1))
        self.self_attn_fusion_cycles = nn.ModuleList()
        for _ in range(self.attn_cycles):
            self.self_attn_fusion_cycles.append(MultiHeadSelfAttentionWithSourcePE(feature_channels, num_heads, use_position_encoding=use_position_encoding, num_blocks=1))

        # ======================== 原有模块3：3组独立Cross-Attention（完全保持不变）========================
        self.cross_attn_vis_cycles = nn.ModuleList()
        for _ in range(self.attn_cycles):
            self.cross_attn_vis_cycles.append(MultiHeadCrossAttentionWithSourcePE(feature_channels, num_heads, use_position_encoding=use_position_encoding, num_blocks=1))
        self.cross_attn_ir_cycles = nn.ModuleList()
        for _ in range(self.attn_cycles):
            self.cross_attn_ir_cycles.append(MultiHeadCrossAttentionWithSourcePE(feature_channels, num_heads, use_position_encoding=use_position_encoding, num_blocks=1))
        self.cross_attn_fusion_vis_cycles = nn.ModuleList()
        for _ in range(self.attn_cycles):
            self.cross_attn_fusion_vis_cycles.append(MultiHeadCrossAttentionWithSourcePE(feature_channels, num_heads, use_position_encoding=use_position_encoding, num_blocks=1))
        self.cross_attn_fusion_ir_cycles = nn.ModuleList()
        for _ in range(self.attn_cycles):
            self.cross_attn_fusion_ir_cycles.append(MultiHeadCrossAttentionWithSourcePE(feature_channels, num_heads, use_position_encoding=use_position_encoding, num_blocks=1))

        # ======================== 原有模块4：3组独立的融合降维卷积（完全保持不变）========================
        self.conv_fusion_cycles = nn.ModuleList()
        for _ in range(self.attn_cycles):
            self.conv_fusion_cycles.append(
                nn.Conv2d(256, 128, kernel_size=1, padding=0, groups=1)  # 保持原卷积参数不变
            )

        # ======================== 原有模块5：Final Layer）========================
        # 分支1 & 2：原始的 Vis 和 Ir 自重建分支（保留）
        self.final_vis_orig = FixedFinalLayer(in_channels=feature_channels * 2, out_channels=vis_img_channels)
        self.final_ir_orig = FixedFinalLayer(in_channels=feature_channels * 2, out_channels=ir_img_channels)

        # 分支3：融合图像生成分支（保留）
        self.final_fusion = FixedFinalLayer(in_channels=feature_channels * 2, out_channels=vis_img_channels)

        # ======================== 原有模块6：损失函数（完全保持不变）========================
        self.compute_loss = Loss(device="cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, img_ir, img_vis, 
                ir_img_full=None, vis_img_full=None, 
                patch_pos=None, img_size=None):
        # 保存原始输入用于位置编码（完全保持不变）
        source_ir = img_ir  # [B,1,H,W]
        source_vis = img_vis  # [B,3,H,W]

        # ---------------------- 步骤1：特征提取（完全保持不变）----------------------
        fir = self.ir_extractor(img_ir)
        fvis = self.vis_extractor(img_vis)
        fvis_concat = torch.cat([fir, fvis], dim=1)
        ffusion = self.fusion_extractor(fvis_concat)
        _, _, h_feat, w_feat = fir.shape

        # ---------------------- 步骤2：三轮self→cross循环迭代（完全保持不变）----------------------
        curr_fvis = fvis
        curr_fir = fir
        curr_ffusion = ffusion

        for cycle in range(self.attn_cycles):
            # 本轮专属自注意力
            curr_fvis, _ = self.self_attn_vis_cycles[cycle](curr_fvis)
            curr_fir, _ = self.self_attn_ir_cycles[cycle](curr_fir)
            curr_ffusion, _ = self.self_attn_fusion_cycles[cycle](curr_ffusion)

            # 本轮专属交叉注意力
            attn_vis_out, _ = self.cross_attn_vis_cycles[cycle](curr_fvis, curr_ffusion, curr_ffusion)
            attn_ir_out, _ = self.cross_attn_ir_cycles[cycle](curr_fir, curr_ffusion, curr_ffusion)
            attn_fusion_out_vis, _ = self.cross_attn_fusion_vis_cycles[cycle](curr_ffusion, curr_fvis, curr_fvis)
            attn_fusion_out_ir, _ = self.cross_attn_fusion_ir_cycles[cycle](curr_ffusion, curr_fir, curr_fir)

            # 特征更新
            curr_fvis = attn_vis_out
            curr_fir = attn_ir_out
            curr_ffusion = torch.cat([attn_fusion_out_vis, attn_fusion_out_ir], dim=1)
            curr_ffusion = self.conv_fusion_cycles[cycle](curr_ffusion)

        # 步骤3 简化为
        vis_exclusive_feature = curr_ffusion - curr_fir
        ir_exclusive_feature  = curr_ffusion - curr_fvis

        # ---------------------- 步骤4：图像重建 ----------------------
        # 分支1 & 2：原始的自重建（保留原有逻辑）
        img_vis_pred_orig = self.final_vis_orig(curr_fvis)
        img_ir_pred_orig = self.final_ir_orig(curr_fir)

        # 分支3：融合图像
        img_fusion_pred = self.final_fusion(curr_ffusion)

        return {
            # 原始自重建输出
            "img_vis_pred": img_vis_pred_orig,
            "img_ir_pred": img_ir_pred_orig,
            # 融合输出
            "img_fusion_pred": img_fusion_pred,
            "features": {
                "fir": curr_fir, 
                "fvis": curr_fvis, 
                "ffusion": curr_ffusion,
                "vis_exclusive_feature": vis_exclusive_feature,
                "ir_exclusive_feature": ir_exclusive_feature,
            }
        }
