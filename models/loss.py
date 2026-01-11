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
            sigma: 高斯核标准差
        Returns:
            blurred: 模糊后的张量 [B, C, H, W]
        """
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
                 # 移除：saliency_root相关配置（不再从本地加载）
                 # 基础损失权重
                 lambda_vis=1.0,
                 lambda_ir=1.0,
                 lambda_perceptual=0,
                 lambda_gradient=0,
                 lambda_style=0,
                 lambda_pvs=0.2,
                 # 拆分后的gradloss/intloss独立权重
                 lambda_gradloss=1.0,
                 lambda_intloss=0,
                 lambda_maxintloss=1.0,
                 # ========== 新增：Color Loss权重 ==========
                 lambda_color=5.0,
                 # GradientLoss相关参数
                 grad_loss_type='l1',
                 grad_reduction='mean'):
        super().__init__()
        self.device = device

        # ========== 损失权重配置（新增lambda_color） ==========
        self.lambda_dict = {
            'vis': lambda_vis,
            'ir': lambda_ir,
            'perceptual': lambda_perceptual,
            'gradient': lambda_gradient,
            'style': lambda_style,
            'pvs': lambda_pvs,
            'gradloss': lambda_gradloss,
            'intloss': lambda_intloss,
            'maxintloss': lambda_maxintloss,
            'color': lambda_color  # 新增color loss权重
        }

        # ========== 梯度损失相关参数 ==========
        self.grad_loss_type = grad_loss_type
        self.grad_reduction = grad_reduction
        # 预定义Sobel算子（仅初始化一次）
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                    dtype=torch.float32, device=device).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                    dtype=torch.float32, device=device).view(1, 1, 3, 3)

        # ========== VGG模型（仅在需要时加载） ==========
        self.vgg = None
        if self.lambda_dict['perceptual'] > 0 or self.lambda_dict['style'] > 0:
            try:
                self.vgg = self._build_vgg().to(device).eval()
                for param in self.vgg.parameters():
                    param.requires_grad = False
            except Exception as e:
                print(f"Warning: VGG加载失败，感知/风格损失将禁用 | 错误：{e}")
                self.lambda_dict['perceptual'] = 0.0
                self.lambda_dict['style'] = 0.0

    # ========== 新增：RGB转YCbCr并计算颜色一致性损失 ==========
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
        计算颜色一致性损失：融合图像与可见光图像的CbCr通道差异
        Args:
            pred_fusion: [B, C, H, W]，融合图像，数值范围[-1,1]
            target_vis: [B, C, H, W]，可见光图像，数值范围[-1,1]
        Returns:
            color_loss: 标量损失值，公式：1/(HW) * ||CbCr(If) - CbCr(Ivis)||1
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

    # ========== 原GradientLoss核心方法（不变） ==========
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

    # ========== VGG构建（不变） ==========
    def _build_vgg(self):
        """构建简化的VGG特征提取器"""
        vgg = torchvision.models.vgg16(pretrained=True).features[:4]
        return vgg

    # ========== 可见光/红外L1损失（不变） ==========
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

    def _compute_intloss(self, pred_fusion, target_vis, target_ir, saliency_tensor):
        """
        直接使用传入的saliency张量计算intloss，不再从本地加载文件
        关键修改：
        1. 入参替换为saliency_tensor（[B, 1, H, W]，范围[0,1]）
        2. 移除批量加载文件的逻辑，直接处理传入的张量
        3. 保留尺寸校验和resize，提升鲁棒性
        :param pred_fusion: 预测融合图像 [B, C, H, W]，范围[-1,1]
        :param target_vis: 目标可见光图像 [B, C, H, W]，范围[-1,1]
        :param target_ir: 目标红外图像 [B, 1, H, W]，范围[-1,1]
        :param saliency_tensor: 显著性张量 [B, 1, H, W]，范围[0,1]（从Dataset同步传入）
        :return: intloss: 标量损失值
        """
        # ========== 1. 权重为0时直接返回0 ==========
        if self.lambda_dict['intloss'] == 0:
            return torch.tensor(0.0, device=self.device)
        
        B, C, H, W = pred_fusion.shape
        
        # ========== 2. 校验并适配saliency张量尺寸 ==========
        SR = saliency_tensor.to(self.device, non_blocking=True)  # 确保设备对齐
        # 校验尺寸：若与输入图像不一致，自适应resize
        if SR.shape[2:] != (H, W):
            SR = F.interpolate(SR, size=(H, W), mode='bilinear', align_corners=False)
        # 确保通道数为1，形状为[B, 1, H, W]
        if SR.shape[1] != 1:
            SR = SR[:, 0:1, :, :]
        
        # ========== 3. 确保saliency张量数值范围为[0,1]（与原逻辑对齐） ==========
        SR = torch.clamp(SR, 0.0, 1.0)
        
        # ========== 4. 计算可见光显著性权重 ==========
        SV = 1 - SR  # [B, 1, H, W]，值范围[0,1]
        
        # ========== 5. 扩展权重到与图像相同的通道数 ==========
        omega_V_expanded = SV.repeat(1, C, 1, 1)  # [B, C, H, W]
        omega_R_expanded = SR.repeat(1, C, 1, 1)  # [B, C, H, W]
        
        # ========== 6. 统一所有图像的数值范围（[-1,1] → [0,1]） ==========
        pred_fusion = (pred_fusion + 1) / 2  # [B,C,H,W] → [0,1]
        target_vis = (target_vis + 1) / 2    # 关键：目标图也要归一化到[0,1]
        target_ir = (target_ir + 1) / 2      # 关键：红外图也要归一化到[0,1]
        
        # ========== 7. 红外图像通道扩展（鲁棒版） ==========
        if target_ir.shape[1] != C:
            target_ir_3ch = target_ir.repeat(1, C, 1, 1)  # [B,C,H,W]
        else:
            target_ir_3ch = target_ir  # 已匹配通道数，无需扩展
        
        # ========== 8. 计算加权L1损失（维度完全对齐） ==========
        loss_vis = F.l1_loss(omega_V_expanded * pred_fusion, omega_V_expanded * target_vis)
        loss_ir = F.l1_loss(omega_R_expanded * pred_fusion, omega_R_expanded * target_ir_3ch)
        intloss = loss_vis + loss_ir
    
        return intloss
    
    def _compute_maxintloss(self, pred_fusion, target_vis, target_ir):
        """
        计算内容保持拆分后的强度损失（maxintloss）
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
        # ========== 1. 统一数值范围：[-1,1] → [0,1]（和intloss保持一致） ==========
        # 消除数值范围导致的误差量级差异，保证权重配置生效
        pred_fusion = (pred_fusion + 1) / 2  # 融合图像归一化
        target_vis = (target_vis + 1) / 2    # 可见光图像归一化
        target_ir = (target_ir + 1) / 2      # 红外图像归一化

        # ========== 2. 动态扩展红外通道数（替换硬编码的3，适配任意通道配置） ==========
        C = pred_fusion.shape[1]  # 获取融合图像的通道数（如3/1）
        # 仅当红外通道数≠融合图像通道数时，才扩展（鲁棒性更强）
        if target_ir.shape[1] != C:
            intensity_ir = target_ir.repeat(1, C, 1, 1)  # [B,1,H,W] → [B,C,H,W]
        else:
            intensity_ir = target_ir  # 已匹配通道数，无需扩展

        # ========== 3. 保留原逻辑：像素级max(可见光, 红外) + 绝对误差计算 ==========
        intensity_fusion = pred_fusion
        intensity_vis = target_vis
        # 像素级取可见光/红外的最大值（保留高频信息）
        max_vis_ir = torch.max(intensity_vis, intensity_ir)
        # 融合图像与max值的像素差异（取绝对值，避免正负抵消）
        pixel_diff = torch.abs(intensity_fusion - max_vis_ir)

        # ========== 4. 归一化损失（除以总像素数，标准化损失值） ==========
        B, _, H, W = pixel_diff.shape  # 忽略通道数（已匹配）
        maxintloss = pixel_diff.sum() / (B * C * H * W)  # 平均像素误差
        
        return maxintloss

    # ========== gradloss计算（不变） ==========
    def _compute_gradloss(self, pred_fusion, target_vis, target_ir):
        # 计算各图像的Sobel梯度（取绝对值）
        grad_fusion = torch.abs(self._sobel_gradient(pred_fusion))
        grad_vis = torch.abs(self._sobel_gradient(target_vis))
        grad_ir = torch.abs(self._sobel_gradient(target_ir))
        # 融合图像梯度需匹配可见光/红外梯度的最大值
        gradloss = F.mse_loss(grad_fusion, torch.max(grad_vis, grad_ir))
        return gradloss

    # ========== 全局梯度损失（不变） ==========
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

    def _compute_perceptual_loss(self, pred_vis, pred_ir, pred_fusion, targets):
        """优化后的感知损失：仅保留fusion分支，删除vis/ir分支"""
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

        # ========== 核心修改：仅计算fusion分支的感知损失 ==========
        # 删掉perceptual_vis和perceptual_ir的计算，只保留fusion
        perceptual_fusion = _single_perceptual(pred_fusion, target_fusion)
        return perceptual_fusion

    # ========== 风格损失（不变） ==========
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

    # ========== PVS损失（不变） ==========
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

    # ========== Sobel梯度计算（不变） ==========
    def _sobel_gradient(self, x):
        """计算图像的Sobel梯度，使用 Replicate Padding 消除边缘黑线"""
        sobel_x = self.sobel_x.repeat(x.shape[1], 1, 1, 1)
        sobel_y = self.sobel_y.repeat(x.shape[1], 1, 1, 1)

        x_padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
        grad_x = F.conv2d(x_padded, sobel_x, padding=0, groups=x.shape[1])
        grad_y = F.conv2d(x_padded, sobel_y, padding=0, groups=x.shape[1])

        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    # ========== 修改后的前向传播（接收saliency张量，不再加载文件） ==========
    def forward(self, outputs, targets, saliency_tensor):
        """
        前向传播：直接使用传入的saliency张量计算损失
        :param outputs: 模型输出字典
        :param targets: 目标数据字典
        :param saliency_tensor: 显著性张量 [B, 1, H, W]，范围[0,1]（从Dataset同步传入）
        :return: 损失字典
        """
        pred_vis = outputs["img_vis_pred"]
        pred_ir = outputs["img_ir_pred"]
        pred_fusion = outputs["img_fusion_pred"]
        target_vis = targets["img_vis"]
        target_ir = targets["img_ir"]

        # 1. 计算基础L1损失
        l1_vis = self._compute_l1_vis(pred_vis, target_vis)
        l1_ir = self._compute_l1_ir(pred_ir, target_ir)

        # 2. 调用修改后的方法计算intloss（传入saliency张量，不再传文件名）
        intloss = self._compute_intloss(pred_fusion, target_vis, target_ir, saliency_tensor)
        maxintloss = self._compute_maxintloss(pred_fusion, target_vis, target_ir)
        
        # 3. 计算gradloss
        gradloss = self._compute_gradloss(pred_fusion, target_vis, target_ir)

        # 4. 计算其他损失项
        grad_loss = self._compute_gradient_loss(pred_vis, pred_ir, pred_fusion, targets)
        perceptual_loss = self._compute_perceptual_loss(pred_vis, pred_ir, pred_fusion, targets)
        style_loss = self._compute_style_loss(pred_fusion, targets)
        pvs_loss = self._compute_pvs_loss(pred_fusion, targets)

        # ========== 新增：计算颜色一致性损失 ==========
        color_loss = self._compute_color_loss(pred_fusion, target_vis)

        # 5. 总损失计算（权重加权求和，新增color loss项）
        total_loss = (
            self.lambda_dict['vis'] * l1_vis +
            self.lambda_dict['ir'] * l1_ir +
            self.lambda_dict['gradloss'] * gradloss +
            self.lambda_dict['intloss'] * intloss +
            self.lambda_dict['maxintloss'] * maxintloss +
            self.lambda_dict['gradient'] * grad_loss +
            self.lambda_dict['perceptual'] * perceptual_loss +
            self.lambda_dict['style'] * style_loss +
            self.lambda_dict['pvs'] * pvs_loss +
            self.lambda_dict['color'] * color_loss  # 新增color loss加权项
        )

        # 6. 返回字典（新增color_loss项）
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
            "color_loss": color_loss,  # 新增返回color loss
            "lambda_config": self.lambda_dict.copy()
        }
