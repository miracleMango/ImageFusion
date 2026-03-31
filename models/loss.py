import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import os
import numpy as np

def gaussian_blur_manual(x, kernel_size, sigma):
        """
        手动实现高斯模糊（适配PyTorch<1.9.0）
        Args:
            x: 输入张量 [B, C, H, W]
            kernel_size: 高斯核尺寸（奇数）
            sigma: 高斯核标准差（标量/列表/元组，统一转为元组）
        Returns:
            blurred: 模糊后的张量 [B, C, H, W]
        """
        # 适配sigma类型：标量→元组（解决索引报错）
        if isinstance(sigma, (int, float)):
            sigma = (sigma, sigma)
        # 适配设备和数据类型
        device = x.device
        dtype = x.dtype
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        kernel_x = torch.arange(kernel_size[0], device=device, dtype=dtype) - kernel_size[0] // 2
        kernel_y = torch.arange(kernel_size[1], device=device, dtype=dtype) - kernel_size[1] // 2
        kernel_x = torch.exp(-kernel_x.pow(2) / (2 * sigma[0] ** 2))
        kernel_y = torch.exp(-kernel_y.pow(2) / (2 * sigma[1] ** 2))
        kernel = kernel_x.unsqueeze(1) * kernel_y.unsqueeze(0)  # 外积生成2D核
        kernel = kernel / kernel.sum()  # 归一化
        
        # 扩展核维度以适配卷积（适配输入通道数）
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, H_k, W_k]
        kernel = kernel.repeat(x.shape[1], 1, 1, 1)  # [C, 1, H_k, W_k]
        
        # 卷积（保持尺寸不变）
        padding = (kernel_size[1] // 2, kernel_size[0] // 2)
        blurred = F.conv2d(x, kernel, padding=padding, groups=x.shape[1])
        return blurred

class Loss(nn.Module):
    def __init__(self,
                 device,
                 # 基础损失权重
                 lambda_vis=1.0,
                 lambda_ir=1.0,
                 lambda_perceptual=0,
                 lambda_gradient=0,
                 lambda_style=0,
                 lambda_pvs=0.2,
                 # 拆分后的gradloss/intloss独立权重
                 lambda_gradloss=1.0,
                 lambda_grad_en=0,
                 lambda_intloss=1.0,
                 lambda_maxintloss=1.0,
                 # 原有颜色损失权重（颜色一致性）
                 lambda_color=0,
                 # 新增：色彩丰富度损失权重
                 lambda_color_rich=0,
                 # 新增：HSV饱和度拉伸损失权重
                 lambda_saturation=0,
                 # 对比度损失权重
                 lambda_contrast=0,
                 # ========== 新增Canny边缘损失：权重配置 ==========
                 lambda_cannyEdge=0,  # Canny边缘损失权重，默认0.1，可按需调整
                 # GradientLoss相关参数
                 grad_loss_type='l1',
                 grad_reduction='mean'):
        super().__init__()
        self.device = device

        # 损失权重配置（添加cannyEdge权重）
        self.lambda_dict = {
            'vis': lambda_vis,
            'ir': lambda_ir,
            'perceptual': lambda_perceptual,
            'gradient': lambda_gradient,
            'style': lambda_style,
            'pvs': lambda_pvs,
            'gradloss': lambda_gradloss,
            'grad_en': lambda_grad_en,
            'intloss': lambda_intloss,
            'maxintloss': lambda_maxintloss,
            'color': lambda_color,
            'color_rich': lambda_color_rich,
            'saturation': lambda_saturation,
            'contrast': lambda_contrast,
            'cannyEdge': lambda_cannyEdge  # 新增：Canny边缘损失权重
        }

        # 梯度损失相关参数
        self.grad_loss_type = grad_loss_type
        self.grad_reduction = grad_reduction
        # 预定义Sobel算子（仅初始化一次，Canny边缘检测复用）
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                    dtype=torch.float32, device=device).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                    dtype=torch.float32, device=device).view(1, 1, 3, 3)

        # VGG模型（仅在需要时加载）
        self.vgg = None
        # 新增：定义VGG归一化的均值/标准差（解决perceptual loss报错）
        self.vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        if self.lambda_dict['perceptual'] > 0 or self.lambda_dict['style'] > 0:
            try:
                self.vgg = self._build_vgg().to(device).eval()
                for param in self.vgg.parameters():
                    param.requires_grad = False
            except Exception as e:
                print(f"Warning: VGG加载失败，感知/风格损失将禁用 | 错误：{e}")
                self.lambda_dict['perceptual'] = 0.0
                self.lambda_dict['style'] = 0.0

    # ========== 新增核心：PyTorch版Canny边缘检测（张量实现，支持GPU/自动求导） ==========
    def _canny_edge_detection(self, x):
        """
        【向量化优化版】PyTorch张量实现Canny边缘检测（无任何Python for循环，GPU并行）
        适配[B, C, H, W]格式，支持3通道/单通道，自动灰度化，数值范围[-1,1]→[0,1]
        Returns:
            edge: [B, 1, H, W]，0-1范围，1为边缘，0为非边缘，支持自动求导/GPU
        """
        device = x.device
        dtype = x.dtype
        B, C, H, W = x.shape

        # 步骤1：灰度化（3通道RGB转单通道，单通道直接复用）
        if C == 3:
            gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        else:
            gray = x[:, 0:1, :, :]  # [B,1,H,W]

        # 步骤2：数值范围转换[-1,1] → [0,1]
        gray = (gray + 1) / 2.0
        # 步骤3：高斯滤波去噪（复用原有高斯模糊，保持逻辑一致）
        gray_blur = gaussian_blur_manual(gray, kernel_size=3, sigma=1.0)  # [B,1,H,W]

        # 步骤4：Sobel梯度计算（复用类内Sobel核，Replicate Padding，保持原有逻辑）
        sobel_x = self.sobel_x.to(dtype)
        sobel_y = self.sobel_y.to(dtype)
        gray_pad = F.pad(gray_blur, (1,1,1,1), mode='replicate')
        grad_x = F.conv2d(gray_pad, sobel_x, padding=0, groups=1)  # [B,1,H,W]
        grad_y = F.conv2d(gray_pad, sobel_y, padding=0, groups=1)  # [B,1,H,W]

        # 步骤5：计算梯度幅值和方向（向量化，无循环）
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)  # 避免根号0，[B,1,H,W]
        grad_dir = torch.atan2(grad_y, grad_x)  # 弧度制，[B,1,H,W]
        grad_dir = torch.rad2deg(grad_dir) % 360  # 转换为[0,360)角度制，向量化取模

        # ==============================================
        # 步骤6：向量化非极大值抑制（NMS）—— 替换原H/W逐像素循环
        # ==============================================
        nms = torch.zeros_like(grad_mag)
        # 1. 量化梯度方向为4个主方向（0°、45°、90°、135°），生成方向掩码（向量化）
        angle = grad_dir
        # 方向0：0°/180°（水平）：0<=θ<22.5 或 157.5<=θ<202.5 或 337.5<=θ<360
        mask0 = ((angle >= 0) & (angle < 22.5)) | ((angle >= 157.5) & (angle < 202.5)) | ((angle >= 337.5) & (angle < 360))
        # 方向1：45°/225°：22.5<=θ<67.5 或 202.5<=θ<247.5
        mask1 = ((angle >= 22.5) & (angle < 67.5)) | ((angle >= 202.5) & (angle < 247.5))
        # 方向2：90°/270°（垂直）：67.5<=θ<112.5 或 247.5<=θ<292.5
        mask2 = ((angle >= 67.5) & (angle < 112.5)) | ((angle >= 247.5) & (angle < 292.5))
        # 方向3：135°/315°：112.5<=θ<157.5 或 292.5<=θ<337.5
        mask3 = ((angle >= 112.5) & (angle < 157.5)) | ((angle >= 292.5) & (angle < 337.5))

        # 2. 向量化获取所有方向的邻域幅值（利用张量切片，批量处理所有像素）
        # 边界像素设为0，避免索引越界（与原循环逻辑一致，仅处理1<=i<=H-2,1<=j<=W-2）
        mag_pad = F.pad(grad_mag, (1,1,1,1), mode='constant', value=0)  # 上下左右各补1圈0，[B,1,H+2,W+2]
        # 方向0（水平）：邻域为(j-1,i)和(j+1,i) → 切片为[:, :, 1:-1, :-2] 和 [:, :, 1:-1, 2:]
        nbr0_1 = mag_pad[:, :, 1:-1, :-2]  # 左邻域
        nbr0_2 = mag_pad[:, :, 1:-1, 2:]   # 右邻域
        # 方向1（45°）：邻域为(i-1,j+1)和(i+1,j-1) → 切片为[:, :, :-2, 2:] 和 [:, :, 2:, :-2]
        nbr1_1 = mag_pad[:, :, :-2, 2:]    # 左上邻域
        nbr1_2 = mag_pad[:, :, 2:, :-2]    # 右下邻域
        # 方向2（垂直）：邻域为(i-1,j)和(i+1,j) → 切片为[:, :, :-2, 1:-1] 和 [:, :, 2:, 1:-1]
        nbr2_1 = mag_pad[:, :, :-2, 1:-1]  # 上邻域
        nbr2_2 = mag_pad[:, :, 2:, 1:-1]   # 下邻域
        # 方向3（135°）：邻域为(i-1,j-1)和(i+1,j+1) → 切片为[:, :, :-2, :-2] 和 [:, :, 2:, 2:]
        nbr3_1 = mag_pad[:, :, :-2, :-2]   # 右上邻域
        nbr3_2 = mag_pad[:, :, 2:, 2:]     # 左下邻域

        # 3. 向量化判断局部最大值（所有方向批量判断，替代逐像素if）
        nms[mask0] = grad_mag[mask0] * ((grad_mag[mask0] >= nbr0_1[mask0]) & (grad_mag[mask0] >= nbr0_2[mask0])).float()
        nms[mask1] = grad_mag[mask1] * ((grad_mag[mask1] >= nbr1_1[mask1]) & (grad_mag[mask1] >= nbr1_2[mask1])).float()
        nms[mask2] = grad_mag[mask2] * ((grad_mag[mask2] >= nbr2_1[mask2]) & (grad_mag[mask2] >= nbr2_2[mask2])).float()
        nms[mask3] = grad_mag[mask3] * ((grad_mag[mask3] >= nbr3_1[mask3]) & (grad_mag[mask3] >= nbr3_2[mask3])).float()

        # ==============================================
        # 步骤7：向量化双阈值处理+边缘连接 —— 替换原8邻域遍历循环
        # 核心：用3x3全1卷积实现8邻域检测，批量判断所有弱边缘是否与强边缘相邻
        # ==============================================
        low_thresh = 0.1
        high_thresh = 0.2
        # 1. 生成强/弱边缘掩码（向量化）
        strong = (nms >= high_thresh).float()  # [B,1,H,W]，强边缘为1，其余为0
        weak = ((nms >= low_thresh) & (nms < high_thresh)).float()   # [B,1,H,W]，弱边缘为1，其余为0

        # 2. 向量化8邻域强边缘检测（3x3全1卷积，膨胀操作）
        # 3x3全1卷积核：卷积后值>0表示该像素8邻域存在强边缘
        kernel_3x3 = torch.ones((1,1,3,3), dtype=dtype, device=device)  # [1,1,3,3]
        strong_pad = F.pad(strong, (1,1,1,1), mode='constant', value=0)  # 补0避免边缘误判
        # 卷积计算每个像素的8邻域强边缘总数（groups=1适配单通道）
        strong_neighbor = F.conv2d(strong_pad, kernel_3x3, padding=0, groups=1)
        # 邻域存在强边缘 → strong_neighbor > 0
        has_strong = (strong_neighbor > 1e-8).float()  # 加小值避免浮点误差

        # 3. 边缘连接：弱边缘且邻域有强边缘 → 转为强边缘（向量化合并）
        edge = strong + weak * has_strong
        # 裁剪数值范围为[0,1]，批次保持一致
        edge = torch.clamp(edge, 0.0, 1.0)

        return edge

    # ========== 新增：Canny边缘损失计算函数 ==========
    def _compute_cannyEdge_loss(self, pred_fusion, target_vis):
        """
        计算Canny边缘损失：fusion边缘图与vis边缘图的L1损失（相减取绝对值求平均）
        Args:
            pred_fusion: 融合图像 [B, C, H, W]，范围[-1,1]
            target_vis: 可见光图像 [B, C, H, W]，范围[-1,1]
        Returns:
            cannyEdge_loss: 标量损失值
        """
        # 权重为0时直接返回0，避免无效计算
        if self.lambda_dict['cannyEdge'] == 0:
            return torch.tensor(0.0, device=self.device)
        # 分别计算融合图和可见光图的Canny边缘图
        fusion_edge = self._canny_edge_detection(pred_fusion)
        vis_edge = self._canny_edge_detection(target_vis)
        # 核心逻辑：L1损失（相减取绝对值，求平均）
        cannyEdge_loss = F.l1_loss(fusion_edge, vis_edge)
        return cannyEdge_loss

    # ========== 原有对比度损失计算函数 ==========
    def _compute_contrast_loss(self, pred_fusion):
        """
        计算方案B的对比度损失 L_std = max(0, C_target - std(I_fused))
        阈值C_target固定为33，与你统计的vis原图/融合图标准差匹配
        Args:
            pred_fusion: [B, C, H, W]，融合图像，数值范围[-1,1]
        Returns:
            contrast_loss: 标量损失值
        """
        # 权重为0时直接返回0，避免无效计算
        if self.lambda_dict['contrast'] == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 1. 数值范围转换：[-1,1] → [0,255]（和你统计标准差的范围对齐）
        img_0_255 = (pred_fusion + 1) / 2 * 255.0  # 还原为0-255灰度范围
        
        # 2. 3通道RGB转灰度图（匹配你之前统计标准差的逻辑）
        if img_0_255.shape[1] == 3:
            # 标准RGB转灰度公式：0.299R + 0.587G + 0.114B
            gray = 0.299 * img_0_255[:, 0:1, :, :] + \
                   0.587 * img_0_255[:, 1:2, :, :] + \
                   0.114 * img_0_255[:, 2:3, :, :]
        else:
            gray = img_0_255  # 单通道图像直接使用
        
        # 3. 计算每个图像的标准差（在H/W维度计算，保留批次维度）
        # dim=[2,3]：对每个批次的每个图像计算H/W维度的标准差 → [B, 1]
        img_std = torch.std(gray, dim=[2, 3], keepdim=False)
        
        # 4. 计算对比度损失：L_std = max(0, 33 - std)，然后求批次平均
        C_target = 33.0  # 你指定的阈值
        contrast_loss_per_batch = torch.clamp(C_target - img_std, min=0.0)  # max(0, ...)
        contrast_loss = torch.mean(contrast_loss_per_batch)  # 批次平均，得到标量
        
        return contrast_loss

    # ========== 原有梯度增强损失 ==========
    def _compute_grad_enhance_loss(self, pred_fusion):
        '''
        优化版：梯度增强损失 L_grad_en = - mean(log(1 + |∇I_fused|²))
        关键修改：
        1. 求和→平均，避免像素数累积导致数值爆炸
        2. 梯度归一化到[0,1]，稳定log项数值范围
        3. 可选：添加缩放系数，进一步平衡损失尺度
        Args:
            pred_fusion: 融合图像张量 [B, C, H, W]
        Returns:
            grad_en_loss: 标量损失值（合理范围：-10 ~ 0）
        '''
        # 权重为0时直接返回0
        if self.lambda_dict['grad_en'] == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 计算融合图像的梯度
        grad_x = self._compute_gradient(pred_fusion, self.sobel_x)
        grad_y = self._compute_gradient(pred_fusion, self.sobel_y)
        
        # 关键修改1：梯度归一化到[0,1]（消除数值范围过大问题）
        grad_x = torch.tanh(grad_x)  # 先压缩梯度值到[-1,1]
        grad_y = torch.tanh(grad_y)
        grad_x = (grad_x + 1) / 2    # 归一化到[0,1]
        grad_y = (grad_y + 1) / 2
        
        # 计算梯度的平方和 |∇I_fused|²
        grad_square = grad_x ** 2 + grad_y ** 2
        
        # 计算log(1 + 梯度平方和)，添加小值避免log(0)
        log_term = torch.log(1 + grad_square + 1e-8)
        
        # 关键修改2：将所有维度的求和改为平均（核心解决数值大的问题）
        # 原逻辑：sum(H/W) → sum(C) → mean(B)
        # 新逻辑：mean(H/W/C) → mean(B)，最终是全局平均
        grad_en_loss = -torch.mean(log_term, dim=[1,2,3])  # [B]：对C/H/W维度平均
        grad_en_loss = torch.mean(grad_en_loss)            # 对批次平均（标量）
        
        # 可选：添加缩放系数，进一步微调损失尺度（根据需求调整）
        scale_factor = 1.0  # 可设为0.1~10之间，平衡与其他损失的尺度
        grad_en_loss = grad_en_loss * scale_factor
        
        return grad_en_loss

    # ========== 原有色彩丰富度损失 ==========
    def _compute_color_richness_loss(self, pred_fusion):
        """
        计算色彩丰富度损失 L_color = - (√(σ_rg²+σ_yb²) + 0.3√(μ_rg²+μ_yb²))
        Args:
            pred_fusion: [B, C, H, W]，融合图像，数值范围[-1,1]
        Returns:
            color_rich_loss: 标量损失值
        """
        # 权重为0时直接返回0
        if self.lambda_dict['color_rich'] == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 仅处理3通道RGB图像
        if pred_fusion.shape[1] != 3:
            return torch.tensor(0.0, device=self.device)
        
        # 1. 归一化到[0,1]（适配色彩空间计算）
        pred_01 = (pred_fusion + 1) / 2.0  # [-1,1] → [0,1]
        
        # 2. 拆分RGB通道
        R = pred_01[:, 0:1, :, :]  # 红通道
        G = pred_01[:, 1:2, :, :]  # 绿通道
        B = pred_01[:, 2:3, :, :]  # 蓝通道
        
        # 3. 计算RG/YB对立通道
        RG = R - G  # 红-绿通道
        YB = (R + G) / 2 - B  # 黄-蓝通道
        
        # 4. 计算标准差(σ)和均值(μ)（在H/W维度计算，保留B/C维度）
        # 标准差：dim=[2,3] → 对每个批次-通道计算全局σ
        sigma_rg = torch.std(RG, dim=[2, 3], keepdim=True)
        sigma_yb = torch.std(YB, dim=[2, 3], keepdim=True)
        
        # 均值：dim=[2,3] → 对每个批次-通道计算全局μ
        mu_rg = torch.mean(RG, dim=[2, 3], keepdim=True)
        mu_yb = torch.mean(YB, dim=[2, 3], keepdim=True)
        
        # 5. 计算色彩丰富度项
        rich_term = torch.sqrt(sigma_rg**2 + sigma_yb**2) + 0.3 * torch.sqrt(mu_rg**2 + mu_yb**2)
        
        # 6. 损失值 = -rich_term（让模型最大化rich_term），取平均得到标量
        color_rich_loss = -torch.mean(rich_term)
        
        return color_rich_loss

    # ========== 原有HSV饱和度损失 ==========
    def _compute_saturation_loss(self, pred_fusion):
        """
        计算HSV饱和度拉伸损失 L_sat = E[(1 - S_fused)²]
        Args:
            pred_fusion: [B, C, H, W]，融合图像，数值范围[-1,1]
        Returns:
            saturation_loss: 标量损失值
        """
        # 权重为0时直接返回0
        if self.lambda_dict['saturation'] == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 仅处理3通道RGB图像
        if pred_fusion.shape[1] != 3:
            return torch.tensor(0.0, device=self.device)
        
        # 1. 归一化到[0,1]（HSV转换要求输入范围[0,1]）
        pred_01 = (pred_fusion + 1) / 2.0  # [-1,1] → [0,1]
        
        # 2. RGB转HSV（使用修复后的函数）
        # 注意：torch.clamp避免数值溢出（比如1.0+1e-8）
        pred_01 = torch.clamp(pred_01, 0.0, 1.0)
        hsv = rgb_to_hsv(pred_01)  # [B, 3, H, W]，H:0~1, S:0~1, V:0~1
        
        # 3. 提取饱和度通道（S通道，索引1）
        S = hsv[:, 1:2, :, :]  # [B, 1, H, W]
        
        # 4. 计算损失：E[(1 - S)²] → 所有像素的平均平方误差
        saturation_loss = torch.mean((1 - S) ** 2)
        
        return saturation_loss

    # ========== 原有RGB转YCbCr并计算颜色一致性损失 ==========
    def _rgb_to_ycbcr(self, rgb_tensor):
        """
        将RGB张量（0-1范围）转换为YCbCr张量
        Args:
            rgb_tensor: [B, 3, H, W]，RGB格式，数值范围0~1
        Returns:
            ycbcr_tensor: [B, 3, H, W]，YCbCr格式，Y:0~1, Cb/Cr:-0.5~0.5
        """
        # RGB转YCbCr的转换矩阵（标准公式）
        transform_matrix = torch.tensor([
            [0.299, 0.587, 0.114],    # Y通道
            [-0.1687, -0.3313, 0.5],  # Cb通道
            [0.5, -0.4187, -0.0813]   # Cr通道
        ], dtype=rgb_tensor.dtype, device=rgb_tensor.device)
        
        # 批量矩阵乘法实现通道转换
        # 调整维度：[B, 3, H, W] → [B, H, W, 3]
        rgb_permuted = rgb_tensor.permute(0, 2, 3, 1)
        # 矩阵乘法：[B, H, W, 3] × [3, 3] → [B, H, W, 3]
        ycbcr_permuted = torch.matmul(rgb_permuted, transform_matrix.t())
        # Cb/Cr通道偏移0.5（使其范围为-0.5~0.5）
        ycbcr_permuted[..., 1:] += 0.5
        # 恢复维度：[B, H, W, 3] → [B, 3, H, W]
        ycbcr_tensor = ycbcr_permuted.permute(0, 3, 1, 2)
        return ycbcr_tensor

    def _compute_color_loss(self, pred_fusion, target_vis):
        """
        计算颜色一致性损失：融合图像与可见光图像的CbCr通道差异（原有逻辑）
        Args:
            pred_fusion: [B, C, H, W]，融合图像，数值范围[-1,1]
            target_vis: [B, C, H, W]，可见光图像，数值范围[-1,1]
        Returns:
            color_loss: 标量损失值
        """
        # 1. 权重为0时直接返回0
        if self.lambda_dict['color'] == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 2. 检查通道数（仅处理3通道RGB图像）
        if pred_fusion.shape[1] != 3 or target_vis.shape[1] != 3:
            return torch.tensor(0.0, device=self.device)
        
        # 3. 将图像从[-1,1]归一化到[0,1]（适配RGB转YCbCr的公式）
        pred_fusion_01 = (pred_fusion + 1) / 2.0
        target_vis_01 = (target_vis + 1) / 2.0
        
        # 4. 转换到YCbCr空间
        fusion_ycbcr = self._rgb_to_ycbcr(pred_fusion_01)
        vis_ycbcr = self._rgb_to_ycbcr(target_vis_01)
        
        # 5. 提取Cb(通道1)和Cr(通道2)，计算L1损失
        cb_loss = F.l1_loss(fusion_ycbcr[:, 1:2, :, :], vis_ycbcr[:, 1:2, :, :], reduction='none')
        cr_loss = F.l1_loss(fusion_ycbcr[:, 2:3, :, :], vis_ycbcr[:, 2:3, :, :], reduction='none')
        
        # 6. 按公式计算平均损失：1/(HW) * 求和（Cb损失 + Cr损失）/2
        B, _, H, W = cb_loss.shape
        total_pixels = B * H * W
        color_loss = (cb_loss.sum() + cr_loss.sum()) / (2 * total_pixels)
        
        return color_loss

    # ========== 原有GradientLoss核心方法 ==========
    def _compute_gradient(self, x, kernel):
        """计算单方向梯度（x/y），使用 Replicate Padding 消除边缘黑线"""
        b, c, h, w = x.shape
        kernel = kernel.repeat(c, 1, 1, 1)
        pad_size = (kernel.size(-1) - 1) // 2
        x_padded = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='replicate')
        gradient = F.conv2d(x_padded, kernel, padding=0, groups=c)
        return gradient

    def _gradient_loss(self, pred, target):
        """计算单组图像的梯度损失（全局梯度损失用）"""
        if pred.size() != target.size():
            raise ValueError(f"尺寸不匹配：{pred.size()} vs {target.size()}")

        pred_grad_x = self._compute_gradient(pred, self.sobel_x)
        pred_grad_y = self._compute_gradient(pred, self.sobel_y)
        target_grad_x = self._compute_gradient(target, self.sobel_x)
        target_grad_y = self._compute_gradient(target, self.sobel_y)

        if self.grad_loss_type == 'l1':
            grad_diff_x = F.l1_loss(pred_grad_x, target_grad_x, reduction='none')
            grad_diff_y = F.l1_loss(pred_grad_y, target_grad_y, reduction='none')
        elif self.grad_loss_type == 'l2':
            grad_diff_x = F.mse_loss(pred_grad_x, target_grad_x, reduction='none')
            grad_diff_y = F.mse_loss(pred_grad_y, target_grad_y, reduction='none')
        else:
            raise ValueError(f"不支持的梯度损失类型: {self.grad_loss_type}")

        gradient_loss = (grad_diff_x + grad_diff_y) / 2
        if self.grad_reduction == 'mean':
            return gradient_loss.mean()
        elif self.grad_reduction == 'sum':
            return gradient_loss.sum()
        elif self.grad_reduction == 'none':
            return gradient_loss
        else:
            raise ValueError(f"不支持的reduction类型: {self.grad_reduction}")

    # ========== 原有VGG构建 ==========
    def _build_vgg(self):
        """构建简化的VGG特征提取器"""
        vgg = torchvision.models.vgg16(pretrained=True).features[:4]
        return vgg

    # ========== 原有可见光/红外L1损失 ==========
    def _compute_l1_vis(self, pred_vis, target_vis):
        """计算可见光图像的L1损失"""
        if self.lambda_dict['vis'] == 0:
            return torch.tensor(0.0, device=self.device)
        return F.l1_loss(pred_vis, target_vis)

    def _compute_l1_ir(self, pred_ir, target_ir):
        """计算红外图像的L1损失"""
        if self.lambda_dict['ir'] == 0:
            return torch.tensor(0.0, device=self.device)
        return F.l1_loss(pred_ir, target_ir)

    # ========== 原有intloss ==========
    def _compute_intloss(self, pred_fusion, target_vis, target_ir):
        """
        计算intloss（原有逻辑）
        :param pred_fusion: 预测融合图像 [B, C, H, W]，范围[-1,1]
        :param target_vis: 目标可见光图像 [B, C, H, W]，范围[-1,1]
        :param target_ir: 目标红外图像 [B, 1, H, W]，范围[-1,1]
        :return: intloss: 标量损失值
        """
        # 1. 基础配置与设备对齐
        device = self.device  # 用类内的device
        # 权重为0时直接返回0（避免无效计算）
        if self.lambda_dict['intloss'] == 0:
            return torch.tensor(0.0, device=device)
        
        B, C, H, W = pred_fusion.shape
        
        # 2. 对target_ir计算显著性图（核心：替换原heat_mask逻辑）
        # 步骤1：处理target_ir的数值范围（[-1,1] → [0,1]），适配显著性计算
        target_ir_01 = (target_ir + 1) / 2.0  # [B, 1, H, W]，范围[0,1]
        
        # 步骤2：实现显著性计算
        # 红外图像是单通道，先重复为3通道（适配高斯模糊逻辑）
        ir_3ch = target_ir_01.repeat(1, 3, 1, 1)  # [B, 3, H, W]
        
        # 关键修改：调用类外的gaussian_blur_manual函数
        blurred = gaussian_blur_manual(ir_3ch, kernel_size=3, sigma=1.0)
        
        # 分离RGB通道
        R = blurred[:, 0:1, :, :]
        G = blurred[:, 1:2, :, :]
        B_channel = blurred[:, 2:3, :, :]
        
        # 计算每个通道的均值（保持维度，便于广播）
        R_mean = R.mean(dim=[2, 3], keepdim=True)
        G_mean = G.mean(dim=[2, 3], keepdim=True)
        B_mean = B_channel.mean(dim=[2, 3], keepdim=True)
        
        # 计算欧氏距离平方（近似显著性）
        saliency = (R - R_mean).pow(2) + (G - G_mean).pow(2) + (B_channel - B_mean).pow(2)
        
        # 归一化到[0, 1]（鲁棒性处理，避免除0）
        saliency_min = saliency.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        saliency_max = saliency.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        saliency_range = saliency_max - saliency_min + 1e-8  # 加小值避免除0
        SR = (saliency - saliency_min) / saliency_range  # 最终的显著性图（SR）
        
        # 3. 显著性图后处理（与原逻辑一致，确保兼容性）
        # 设备对齐 + 克隆避免原张量被修改
        SR = SR.clone().to(device, non_blocking=True)
        
        # 校验尺寸：若与输入图像不一致，自适应resize
        if SR.shape[2:] != (H, W):
            SR = F.interpolate(SR, size=(H, W), mode='bilinear', align_corners=False)
        # 确保通道数为1（兼容异常输入）
        if SR.shape[1] != 1:
            SR = SR[:, 0:1, :, :]
        # 最终校验数值范围（确保0-1，鲁棒性处理）
        SR = torch.clamp(SR, 0.0, 1.0)
        
        # 4. 原有损失计算逻辑（完全保留，无修改）
        # 计算可见光显著性权重
        SV = 1 - SR  # [B, 1, H, W]
        
        # 扩展权重到与图像相同的通道数
        omega_V_expanded = SV.repeat(1, C, 1, 1)  # [B, C, H, W]
        omega_R_expanded = SR.repeat(1, C, 1, 1)  # [B, C, H, W]
        
        # 统一所有图像的数值范围（[-1,1] → [0,1]）
        pred_fusion = (pred_fusion + 1) / 2
        target_vis = (target_vis + 1) / 2
        target_ir = (target_ir + 1) / 2  # 重新处理target_ir的范围，用于损失计算
        
        # 红外图像通道扩展（鲁棒版）
        if target_ir.shape[1] != C:
            target_ir_3ch = target_ir.repeat(1, C, 1, 1)
        else:
            target_ir_3ch = target_ir
        
        # 计算加权L1损失
        loss_vis = F.l1_loss(omega_V_expanded * pred_fusion, omega_V_expanded * target_vis)
        loss_ir = F.l1_loss(omega_R_expanded * pred_fusion, omega_R_expanded * target_ir_3ch)
        intloss = loss_vis + loss_ir

        return intloss
    
    # ========== 原有maxintloss ==========
    def _compute_maxintloss(self, pred_fusion, target_vis, target_ir):
        """
        计算内容保持拆分后的强度损失（maxintloss）（原有逻辑）
        关键修改：
        1. 统一所有图像到[0,1]范围（和intloss对齐，避免误差量级差异）
        2. 动态扩展红外通道数（替换硬编码的3，适配不同通道配置）
        参数：
            pred_fusion: 预测的融合图像 [B, C, H, W]，数值范围[-1,1]
            target_vis: 目标可见光图像 [B, C, H, W]，数值范围[-1,1]
            target_ir: 目标红外图像 [B, 1, H, W]，数值范围[-1,1]
        返回：
            maxintloss: 强度匹配损失值（标量）
        """
        # 1. 统一数值范围：[-1,1] → [0,1]（和intloss保持一致）
        pred_fusion = (pred_fusion + 1) / 2  # 融合图像归一化
        target_vis = (target_vis + 1) / 2    # 可见光图像归一化
        target_ir = (target_ir + 1) / 2      # 红外图像归一化

        # 2. 动态扩展红外通道数（替换硬编码的3，适配任意通道配置）
        C = pred_fusion.shape[1]  # 获取融合图像的通道数（如3/1）
        # 仅当红外通道数≠融合图像通道数时，才扩展（鲁棒性更强）
        if target_ir.shape[1] != C:
            intensity_ir = target_ir.repeat(1, C, 1, 1)  # [B,1,H,W] → [B,C,H,W]
        else:
            intensity_ir = target_ir  # 已匹配通道数，无需扩展

        # 3. 保留原逻辑：像素级max(可见光, 红外) + 绝对误差计算
        intensity_fusion = pred_fusion
        intensity_vis = target_vis
        # 像素级取可见光/红外的最大值（保留高频信息）
        max_vis_ir = torch.max(intensity_vis, intensity_ir)
        # 融合图像与max值的像素差异（取绝对值，避免正负抵消）
        pixel_diff = torch.abs(intensity_fusion - max_vis_ir)

        # 4. 归一化损失（除以总像素数，标准化损失值）
        B, _, H, W = pixel_diff.shape  # 忽略通道数（已匹配）
        maxintloss = pixel_diff.sum() / (B * C * H * W)  # 平均像素误差
        
        return maxintloss

    # ========== 原有gradloss计算 ==========
    def _compute_gradloss(self, pred_fusion, target_vis, target_ir):
        # 计算各图像的Sobel梯度（取绝对值）
        grad_fusion = torch.abs(self._sobel_gradient(pred_fusion))
        grad_vis = torch.abs(self._sobel_gradient(target_vis))
        grad_ir = torch.abs(self._sobel_gradient(target_ir))
        # 融合图像梯度需匹配可见光/红外梯度的最大值
        gradloss = F.mse_loss(grad_fusion, torch.max(grad_vis, grad_ir))
        return gradloss

    # ========== 原有全局梯度损失 ==========
    def _compute_gradient_loss(self, pred_vis, pred_ir, pred_fusion, targets):
        """计算全局梯度损失"""
        if self.lambda_dict['gradient'] == 0:
            return torch.tensor(0.0, device=self.device)

        target_vis = targets["img_vis"]
        target_ir = targets["img_ir"]
        target_fusion = 0.5 * (target_vis + target_ir)

        grad_loss_vis = self._gradient_loss(pred_vis, target_vis)
        grad_loss_ir = self._gradient_loss(pred_ir, target_ir)
        grad_loss_fusion = self._gradient_loss(pred_fusion, target_fusion)
        return grad_loss_vis + grad_loss_ir + grad_loss_fusion

    # ========== 原有感知损失 ==========
    def _compute_perceptual_loss(self, pred_vis, pred_ir, pred_fusion, targets):
        """优化后的感知损失：仅保留fusion分支，删除vis/ir分支（原有逻辑）"""
        if self.lambda_dict['perceptual'] == 0 or self.vgg is None:
            return torch.tensor(0.0, device=self.device)

        target_vis = targets["img_vis"]
        # 原逻辑中target_fusion直接复用target_vis（无监督下的伪参考）
        target_fusion = target_vis  

        # 复用之前优化的_extract_vgg_feat函数（带no_grad+缓存）
        def _single_perceptual(pred, target):
            if pred.shape[1] != 3:
                return torch.tensor(0.0, device=self.device)
            pred_norm = (pred - self.vgg_mean) / self.vgg_std
            target_norm = (target - self.vgg_mean) / self.vgg_std
            with torch.no_grad():  # 关键：关闭梯度，避免冗余计算
                pred_feat = self.vgg(pred_norm)
                target_feat = self.vgg(target_norm)
            return F.l1_loss(pred_feat, target_feat)

        # 核心修改：仅计算fusion分支的感知损失
        perceptual_fusion = _single_perceptual(pred_fusion, target_fusion)
        return perceptual_fusion

    # ========== 原有风格损失 ==========
    def _compute_style_loss(self, pred_fusion, targets):
        """计算风格损失"""
        if self.lambda_dict['style'] == 0 or self.vgg is None:
            return torch.tensor(0.0, device=self.device)

        target_vis = targets["img_vis"]
        target_ir = targets["img_ir"]

        def gram_matrix(x):
            b, c, h, w = x.size()
            features = x.view(b, c, h * w)
            gram = torch.bmm(features, features.transpose(1, 2))
            return gram / (c * h * w)

        mean = torch.tensor([0.485, 0.485, 0.485]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        def _single_style(pred, target):
            pred_norm = (pred - mean) / std
            target_norm = (target - mean) / std
            pred_feat = self.vgg(pred_norm)
            target_feat = self.vgg(target_norm)
            pred_gram = gram_matrix(pred_feat)
            target_gram = gram_matrix(target_feat)
            return F.mse_loss(pred_gram, target_gram)

        return _single_style(pred_fusion, target_vis) + _single_style(pred_fusion, target_ir)

    # ========== 原有PVS损失 ==========
    def _compute_pvs_loss(self, pred_fusion, targets):
        """计算PVS损失"""
        if self.lambda_dict['pvs'] == 0:
            return torch.tensor(0.0, device=self.device)

        target_vis = targets["img_vis"]
        target_ir = targets["img_ir"]
        I_r = target_vis + target_ir - pred_fusion
        I_r = torch.clamp(I_r, -1.0, 1.0)

        grad_If = self._sobel_gradient(pred_fusion)
        B, C, H, W = grad_If.shape
        g_If = (grad_If ** 2).sum(dim=[1, 2, 3]) / (H * W)
        g_If = g_If.mean()

        grad_Ir = self._sobel_gradient(I_r)
        g_Ir = (grad_Ir ** 2).sum(dim=[1, 2, 3]) / (H * W)
        g_Ir = g_Ir.mean()

        return g_Ir / (g_If + 1e-8)

    # ========== 原有Sobel梯度计算 ==========
    def _sobel_gradient(self, x):
        """计算图像的Sobel梯度，使用 Replicate Padding 消除边缘黑线"""
        sobel_x = self.sobel_x.repeat(x.shape[1], 1, 1, 1)
        sobel_y = self.sobel_y.repeat(x.shape[1], 1, 1, 1)

        x_padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
        grad_x = F.conv2d(x_padded, sobel_x, padding=0, groups=x.shape[1])
        grad_y = F.conv2d(x_padded, sobel_y, padding=0, groups=x.shape[1])

        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    # ========== 修改后前向传播：集成Canny边缘损失 ==========
    def forward(self, outputs, targets):
        """
        前向传播：集成Canny边缘损失、色彩丰富度损失、HSV饱和度损失、对比度损失
        :param outputs: 模型输出字典
        :param targets: 目标数据字典
        :return: 损失字典
        """
        pred_vis = outputs["img_vis_pred"]
        pred_ir = outputs["img_ir_pred"]
        pred_fusion = outputs["img_fusion_pred"]
        target_vis = targets["img_vis"]
        target_ir = targets["img_ir"]
        #pred_vis_decouple = outputs["img_vis_pred_decoup"]
        #pred_ir_decouple = outputs["img_ir_pred_decoup"]

        # 1. 计算基础L1损失
        l1_vis = self._compute_l1_vis(pred_vis, target_vis)
        l1_ir = self._compute_l1_ir(pred_ir, target_ir)

        #l1_vis_decouple = self._compute_l1_vis(pred_vis_decouple, target_vis)
        #l1_ir_decouple = self._compute_l1_ir(pred_ir_decouple, target_ir)

        # 2. 调用修正后的intloss和maxintloss
        intloss = self._compute_intloss(pred_fusion, target_vis, target_ir)
        maxintloss = self._compute_maxintloss(pred_fusion, target_vis, target_ir)
        
        # 3. 计算gradloss
        gradloss = self._compute_gradloss(pred_fusion, target_vis, target_ir)

        # 4. 计算其他损失项
        grad_loss = self._compute_gradient_loss(pred_vis, pred_ir, pred_fusion, targets)
        perceptual_loss = self._compute_perceptual_loss(pred_vis, pred_ir, pred_fusion, targets)
        style_loss = self._compute_style_loss(pred_fusion, targets)
        pvs_loss = self._compute_pvs_loss(pred_fusion, targets)

        # 5. 原有颜色一致性损失
        color_loss = self._compute_color_loss(pred_fusion, target_vis)
        grad_en_loss = self._compute_grad_enhance_loss(pred_fusion)
        
        # 6. 色彩丰富度损失和HSV饱和度损失
        color_rich_loss = self._compute_color_richness_loss(pred_fusion)
        saturation_loss = self._compute_saturation_loss(pred_fusion)

        # 7. 对比度损失
        contrast_loss = self._compute_contrast_loss(pred_fusion)

        # ========== 新增：计算Canny边缘损失 ==========
        cannyEdge_loss = self._compute_cannyEdge_loss(pred_fusion, target_vis)

        # 8. 总损失计算（添加Canny边缘损失）
        total_loss = (
            self.lambda_dict['vis'] * l1_vis +
            self.lambda_dict['ir'] * l1_ir +
            #self.lambda_dict['vis'] * l1_vis_decouple +
            #self.lambda_dict['ir'] * l1_ir_decouple +
            self.lambda_dict['gradloss'] * gradloss +
            self.lambda_dict['intloss'] * intloss +
            self.lambda_dict['maxintloss'] * maxintloss +
            self.lambda_dict['gradient'] * grad_loss +
            self.lambda_dict['perceptual'] * perceptual_loss +
            self.lambda_dict['style'] * style_loss +
            self.lambda_dict['pvs'] * pvs_loss +
            self.lambda_dict['grad_en'] * grad_en_loss +
            self.lambda_dict['color'] * color_loss +
            self.lambda_dict['color_rich'] * color_rich_loss +
            self.lambda_dict['saturation'] * saturation_loss +
            self.lambda_dict['contrast'] * contrast_loss +
            self.lambda_dict['cannyEdge'] * cannyEdge_loss  # 新增：Canny边缘损失项
        )

        # 9. 返回字典（添加Canny边缘损失项）
        return {
            "total_loss": total_loss,
            "l1_vis": l1_vis,
            "l1_ir": l1_ir,
            "grad_loss": grad_loss,
            "perceptual_loss": perceptual_loss,
            "style_loss": style_loss,
            "pvs_loss": pvs_loss,
            "gradloss": gradloss,
            "intloss": intloss,
            "maxintloss": maxintloss,
            "color_loss": color_loss,
            "color_rich_loss": color_rich_loss,
            "saturation_loss": saturation_loss,
            "grad_en_loss": grad_en_loss,
            "contrast_loss": contrast_loss,
            "cannyEdge_loss": cannyEdge_loss,  # 新增：返回Canny边缘损失
            "lambda_config": self.lambda_dict.copy()
        }

# ========== 原有修复后的RGB转HSV函数 ==========
def rgb_to_hsv(rgb):
    """
    修复索引错误的RGB转HSV函数
    Args:
        rgb: [B, 3, H, W]，RGB格式，数值范围0~1
    Returns:
        hsv: [B, 3, H, W]，HSV格式，H:0~1, S:0~1, V:0~1
    """
    # 调整维度：[B, 3, H, W] → [B, H, W, 3]
    rgb = rgb.permute(0, 2, 3, 1)
    B, H, W, C = rgb.shape
    
    # 计算max/min和差值
    max_rgb, argmax_rgb = torch.max(rgb, dim=-1)
    min_rgb, _ = torch.min(rgb, dim=-1)
    delta = max_rgb - min_rgb
    
    # 初始化Hue通道
    h = torch.zeros_like(max_rgb)
    
    # ========== 修复核心：使用高级索引 ==========
    # 1. 生成所有维度的索引
    idx_b, idx_h, idx_w = torch.meshgrid(
        torch.arange(B, device=rgb.device),
        torch.arange(H, device=rgb.device),
        torch.arange(W, device=rgb.device),
        indexing='ij'
    )
    
    # 2. 红色主导（argmax=0）且delta>0
    mask_r = (argmax_rgb == 0) & (delta > 1e-8)
    if mask_r.any():
        # 提取满足条件的像素的RGB值
        r = rgb[idx_b[mask_r], idx_h[mask_r], idx_w[mask_r], 0]
        g = rgb[idx_b[mask_r], idx_h[mask_r], idx_w[mask_r], 1]
        b = rgb[idx_b[mask_r], idx_h[mask_r], idx_w[mask_r], 2]
        d = delta[mask_r]
        # 计算Hue值
        h[mask_r] = ((g - b) / d) % 6
    
    # 3. 绿色主导（argmax=1）且delta>0
    mask_g = (argmax_rgb == 1) & (delta > 1e-8)
    if mask_g.any():
        r = rgb[idx_b[mask_g], idx_h[mask_g], idx_w[mask_g], 0]
        g = rgb[idx_b[mask_g], idx_h[mask_g], idx_w[mask_g], 1]
        b = rgb[idx_b[mask_g], idx_h[mask_g], idx_w[mask_g], 2]
        d = delta[mask_g]
        h[mask_g] = ((b - r) / d) + 2
    
    # 4. 蓝色主导（argmax=2）且delta>0
    mask_b = (argmax_rgb == 2) & (delta > 1e-8)
    if mask_b.any():
        r = rgb[idx_b[mask_b], idx_h[mask_b], idx_w[mask_b], 0]
        g = rgb[idx_b[mask_b], idx_h[mask_b], idx_w[mask_b], 1]
        b = rgb[idx_b[mask_b], idx_h[mask_b], idx_w[mask_b], 2]
        d = delta[mask_b]
        h[mask_b] = ((r - g) / d) + 4
    
    # 归一化Hue到0~1
    h = h / 6.0
    
    # 计算Saturation（饱和度）
    s = torch.where(max_rgb > 1e-8, delta / max_rgb, torch.zeros_like(max_rgb))
    
    # 计算Value（明度）
    v = max_rgb
    
    # 合并并恢复维度：[B, H, W, 3] → [B, 3, H, W]
    hsv = torch.stack([h, s, v], dim=-1).permute(0, 3, 1, 2)
    
    return hsv
