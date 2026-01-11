import torch
import torch.nn as nn
import torch.nn.functional as F
from models.FeatureExtractor import CNNFeatureExtractor, FusionFeatureExtractor
from models.FinalLayer import FixedFinalLayer
from models.crossAttn import MultiHeadCrossAttentionWithSourcePE, MultiHeadSelfAttentionWithSourcePE
from models.loss import Loss

class ImageFusionNetworkWithSourcePE(nn.Module):
    def __init__(self, vis_img_channels=3, ir_img_channels=1, feature_channels=64, num_heads=8,
                 use_position_encoding=True):
        super().__init__()
        # 1. 特征提取器
        self.ir_extractor = CNNFeatureExtractor(ir_img_channels, feature_channels)
        self.vis_extractor = CNNFeatureExtractor(vis_img_channels, feature_channels)
        self.fusion_extractor = FusionFeatureExtractor(feature_channels * 4, feature_channels)  # 恢复原代码*4

        # 2. vis重建分支自注意力模块
        self.self_attn_vis = MultiHeadSelfAttentionWithSourcePE(
            feature_channels, num_heads, use_position_encoding=use_position_encoding,
            src_channels=vis_img_channels, num_blocks=1)

        # 3. ir重建分支自注意力模块
        self.self_attn_ir = MultiHeadSelfAttentionWithSourcePE(
            feature_channels, num_heads, use_position_encoding=use_position_encoding,
            src_channels=vis_img_channels, num_blocks=1)  # 恢复原代码：IR分支也用VIS的src_channels

        # 4. 融合分支自注意力模块
        self.self_attn_fusion = MultiHeadSelfAttentionWithSourcePE(
            feature_channels, num_heads, use_position_encoding=use_position_encoding,
            src_channels=vis_img_channels, num_blocks=1)

        # 5. vis重建分支交叉注意力模块
        self.cross_attn_vis = MultiHeadCrossAttentionWithSourcePE(
            feature_channels, num_heads, use_position_encoding=use_position_encoding,
            src_q_channels=vis_img_channels, src_k_channels=vis_img_channels, src_v_channels=vis_img_channels, num_blocks=1
        )

        # 6. ir重建分支交叉注意力模块
        self.cross_attn_ir = MultiHeadCrossAttentionWithSourcePE(
            feature_channels, num_heads, use_position_encoding=use_position_encoding,
            src_q_channels=vis_img_channels, src_k_channels=vis_img_channels, src_v_channels=vis_img_channels, num_blocks=1
        )

        # 7. 融合分支的交叉注意力模块1
        self.cross_attn_fusion_vis = MultiHeadCrossAttentionWithSourcePE(
            feature_channels, num_heads, use_position_encoding=use_position_encoding,
            src_q_channels=vis_img_channels, src_k_channels=vis_img_channels, src_v_channels=vis_img_channels, num_blocks=1
        )

        # 8. 融合分支的交叉注意力模块2
        self.cross_attn_fusion_ir = MultiHeadCrossAttentionWithSourcePE(
            feature_channels, num_heads, use_position_encoding=use_position_encoding,
            src_q_channels=vis_img_channels, src_k_channels=vis_img_channels, src_v_channels=vis_img_channels, num_blocks=1
        )

        # 9.融合结果通道数降维
        self.conv_fusion = nn.Conv2d(256, 128, kernel_size=1, padding=0, groups=1)

        # 10. Final Layer
        self.final_vis = FixedFinalLayer(
            in_channels=feature_channels * 2,
            out_channels=vis_img_channels
        )
        self.final_ir = FixedFinalLayer(
            in_channels=feature_channels * 2,
            out_channels=ir_img_channels
        )
        self.final_fusion = FixedFinalLayer(
            in_channels=feature_channels * 2,  # 恢复原代码*4
            out_channels=vis_img_channels
        )

        # 11. 损失函数
        self.compute_loss = Loss(device="cuda" if torch.cuda.is_available() else "cpu")

        # 新增：循环轮数（三轮self+cross，核心修改点1）
        self.attn_cycles = 3

    def forward(self, img_ir, img_vis, 
                ir_img_full=None, vis_img_full=None, 
                patch_pos=None, img_size=None):
        """
        完全保留原代码输入参数，仅新增循环逻辑，无其他修改
        """
        # 保存原始输入用于位置编码
        source_ir = img_ir  # [B,1,H,W]
        source_vis = img_vis  # [B,3,H,W]

        # ---------------------- 新增：全局特征 + 位置编码 ----------------------
        #B, _, Hp, Wp = img_ir.shape
        #ir_global_pos_feat = torch.zeros(B, self.pos_encoder.global_dim, device=img_ir.device)
        #vis_global_pos_feat = torch.zeros(B, self.pos_encoder.global_dim, device=img_ir.device)
        
        #if ir_img_full is not None and vis_img_full is not None and patch_pos is not None and img_size is not None:
        #    ir_global_feat = self.ir_global_extractor(ir_img_full)
        #    vis_global_feat = self.vis_global_extractor(vis_img_full)
        #    ir_global_pos_feat = self.pos_encoder(ir_global_feat, patch_pos, img_size)
        #    vis_global_pos_feat = self.pos_encoder(vis_global_feat, patch_pos, img_size)

        # ---------------------- 特征提取 ----------------------
        fir = self.ir_extractor(img_ir)
        fvis = self.vis_extractor(img_vis)

        # 特征拼接
        fvis_concat = torch.cat([fir, fvis], dim=1)
        ffusion = self.fusion_extractor(fvis_concat)

        # 获取特征图尺寸用于源图像下采样
        _, _, h_feat, w_feat = fir.shape

        # 下采样源图像到特征图尺寸
        source_ir_down = F.interpolate(source_ir, size=(h_feat, w_feat), mode='bilinear',
                                       align_corners=False)  # [B,1,h,w]
        source_vis_down = F.interpolate(source_vis, size=(h_feat, w_feat), mode='bilinear',
                                        align_corners=False)  # [B,3,h,w]

        # 融合分支qk的源图像：拼接（1+3=4通道）
        source_fusion_kv = torch.cat([source_ir_down, source_vis_down], dim=1)  # [B,4,h,w]

        # ---------------------- 核心修改：三轮self→cross循环迭代 ----------------------
        # 初始化迭代特征（核心修改点3：用原始特征初始化迭代变量）
        curr_fvis = fvis
        curr_fir = fir
        curr_ffusion = ffusion

        for cycle in range(self.attn_cycles):
            # ========== 第1步：执行本轮自注意力（selfattn），更新特征 ==========
            # vis自注意力：完全恢复原代码输入（source_vis_down）
            curr_fvis, _ = self.self_attn_vis(curr_fvis, source_vis_down)
            # ir自注意力：完全恢复原代码输入（source_vis_down，关键！与模态无关）
            curr_fir, _ = self.self_attn_ir(curr_fir, source_vis_down)
            # 融合分支自注意力：完全恢复原代码输入（source_vis_down，关键！与模态无关）
            curr_ffusion, _ = self.self_attn_fusion(curr_ffusion, source_vis_down)

            # ========== 第2步：执行本轮交叉注意力（crossattn），更新特征 ==========
            # 1. VIS分支交叉注意力：完全恢复原代码输入（source_vis_down * 3，与模态无关）
            attn_vis_out, _ = self.cross_attn_vis(
                curr_fvis, curr_ffusion, curr_ffusion,
                source_vis_down, source_vis_down, source_vis_down
            )
            # 2. IR分支交叉注意力：完全恢复原代码输入（source_vis_down * 3，关键！与模态无关）
            attn_ir_out, _ = self.cross_attn_ir(
                curr_fir, curr_ffusion, curr_ffusion,
                source_vis_down, source_vis_down, source_vis_down
            )
            # 3. 融合分支交叉注意力1：完全恢复原代码输入（source_vis_down * 3）
            attn_fusion_out_vis, _ = self.cross_attn_fusion_vis(
                curr_ffusion, curr_fvis, curr_fvis,
                source_vis_down, source_vis_down, source_vis_down
            )
            # 4. 融合分支交叉注意力2：完全恢复原代码输入（source_vis_down * 3）
            attn_fusion_out_ir, _ = self.cross_attn_fusion_ir(
                curr_ffusion, curr_fir, curr_fir,
                source_vis_down, source_vis_down, source_vis_down
            )

            # ========== 特征更新：本轮交叉注意力输出作为下一轮自注意力输入 ==========
            # 更新VIS/IR特征为交叉注意力输出
            curr_fvis = attn_vis_out
            curr_fir = attn_ir_out
            # 更新融合特征为两个交叉注意力输出的拼接（完全恢复原代码逻辑）
            curr_ffusion = torch.cat([attn_fusion_out_vis, attn_fusion_out_ir], dim=1)
            # 拼接融合结果降维
            curr_ffusion = self.conv_fusion(curr_ffusion)

        # ---------------------- 循环结束：使用最终迭代后的特征继续原逻辑 ----------------------
        # VIS重建
        img_vis_pred = self.final_vis(curr_fvis)

        # IR重建
        img_ir_pred = self.final_ir(curr_fir)

        # 融合注意力输出
        ffusion = curr_ffusion  # 循环后的融合特征

        # 融合图像重建
        img_fusion_pred = self.final_fusion(ffusion)

        return {
            "img_vis_pred": img_vis_pred,
            "img_ir_pred": img_ir_pred,
            "img_fusion_pred": img_fusion_pred,
            "features": {"fir": curr_fir, "fvis": curr_fvis, "ffusion": curr_ffusion}
        }
