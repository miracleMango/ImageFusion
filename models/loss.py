import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Loss(nn.Module):  # 合并后的统一损失类
    def __init__(self,
                 device,
                 # 所有损失项的权重（默认值与原代码一致，可一键修改/置0）
                 lambda_vis=1.0,
                 lambda_ir=1.0,
                 lambda_fusion=1.0,
                 lambda_perceptual=0,
                 lambda_gradient=0,
                 lambda_style=0,
                 lambda_pvs=1.0,
                 # GradientLoss相关参数（整合进来）
                 grad_loss_type='l1',
                 grad_reduction='mean'):
        super().__init__()
        self.device = device

        # ========== 1. 损失权重配置 ==========
        self.lambda_dict = {
            'vis': lambda_vis,
            'ir': lambda_ir,
            'fusion': lambda_fusion,
            'perceptual': lambda_perceptual,
            'gradient': lambda_gradient,
            'style': lambda_style,
            'pvs': lambda_pvs
        }

        # ========== 2. 梯度损失相关参数（原GradientLoss的初始化参数） ==========
        self.grad_loss_type = grad_loss_type
        self.grad_reduction = grad_reduction
        # 预定义Sobel算子（移到此处，仅初始化一次）
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                    dtype=torch.float32, device=device).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                    dtype=torch.float32, device=device).view(1, 1, 3, 3)

        # ========== 3. VGG模型（仅在需要时加载） ==========
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

    # ========== 原GradientLoss的核心方法（整合为内部方法） ==========
    def _compute_gradient(self, x, kernel):
        """原GradientLoss的_compute_gradient方法"""
        b, c, h, w = x.shape
        kernel = kernel.repeat(c, 1, 1, 1)  # [C, 1, 3, 3]
        padding = (kernel.size(-1) - 1) // 2
        gradient = F.conv2d(x, kernel, padding=padding, groups=c)
        return gradient

    def _gradient_loss(self, pred, target):
        """原GradientLoss的forward方法，计算单组图像的梯度损失"""
        if pred.size() != target.size():
            raise ValueError(f"预测图像尺寸 {pred.size()} 与目标图像尺寸 {target.size()} 不匹配")

        # 计算预测/目标图像的梯度
        pred_grad_x = self._compute_gradient(pred, self.sobel_x)
        pred_grad_y = self._compute_gradient(pred, self.sobel_y)
        target_grad_x = self._compute_gradient(target, self.sobel_x)
        target_grad_y = self._compute_gradient(target, self.sobel_y)

        # 计算梯度差异
        if self.grad_loss_type == 'l1':
            grad_diff_x = F.l1_loss(pred_grad_x, target_grad_x, reduction='none')
            grad_diff_y = F.l1_loss(pred_grad_y, target_grad_y, reduction='none')
        elif self.grad_loss_type == 'l2':
            grad_diff_x = F.mse_loss(pred_grad_x, target_grad_x, reduction='none')
            grad_diff_y = F.mse_loss(pred_grad_y, target_grad_y, reduction='none')
        else:
            raise ValueError(f"不支持的梯度损失类型: {self.grad_loss_type}")

        # 合并x/y方向损失并应用reduction
        gradient_loss = (grad_diff_x + grad_diff_y) / 2
        if self.grad_reduction == 'mean':
            return gradient_loss.mean()
        elif self.grad_reduction == 'sum':
            return gradient_loss.sum()
        elif self.grad_reduction == 'none':
            return gradient_loss
        else:
            raise ValueError(f"不支持的reduction类型: {self.grad_reduction}")

    # ========== 原Loss类的核心方法 ==========
    def _build_vgg(self):
        """简化的VGG特征提取器"""
        vgg = torchvision.models.vgg16(pretrained=True).features[:4]
        return vgg

    def _compute_l1_vis(self, pred_vis, target_vis):
        """可见光L1损失"""
        if self.lambda_dict['vis'] == 0:
            return torch.tensor(0.0, device=self.device)
        return F.l1_loss(pred_vis, target_vis)

    def _compute_l1_ir(self, pred_ir, target_ir):
        """红外L1损失"""
        if self.lambda_dict['ir'] == 0:
            return torch.tensor(0.0, device=self.device)
        return F.l1_loss(pred_ir, target_ir)

    def _compute_l1_fusion(self, pred_fusion, targets):
        """融合L1/内容保持损失"""
        if self.lambda_dict['fusion'] == 0:
            return torch.tensor(0.0, device=self.device)

        target_vis = targets["img_vis"]
        target_ir = targets["img_ir"]
        if "img_fusion" in targets:
            return F.l1_loss(pred_fusion, targets["img_fusion"])
        else:
            return self._content_preservation_loss(pred_fusion, target_vis, target_ir)

    def _content_preservation_loss(self, fusion, vis, ir):
        """改进的内容保持损失"""
        grad_fusion = torch.abs(self._sobel_gradient(fusion))
        grad_vis = torch.abs(self._sobel_gradient(vis))
        grad_ir = torch.abs(self._sobel_gradient(ir))
        loss_grad = F.mse_loss(grad_fusion, torch.max(grad_vis, grad_ir))
        return loss_grad

    def _adaptive_intensity_loss(self, fusion, vis, ir, threshold=0.2,
                                 dark_threshold_mode='global', custom_dark_threshold=0.05):
        """自适应强度损失（原逻辑保留）"""
        intensity_fusion = fusion
        intensity_vis = vis
        intensity_ir = ir.repeat(1, 3, 1, 1)

        ir_min = intensity_ir.min()
        ir_max = intensity_ir.max()
        ir_normalized = (intensity_ir - ir_min) / (ir_max - ir_min + 1e-8)

        pixel_weights = torch.sigmoid((ir_normalized - threshold) * 18)

        patch_ir_max_w = torch.max(intensity_ir, dim=3, keepdim=True)[0]
        patch_ir_max = torch.max(patch_ir_max_w, dim=2, keepdim=True)[0]
        patch_ir_max = torch.mean(patch_ir_max, dim=1, keepdim=True)

        if dark_threshold_mode == 'global':
            global_ir_mean = torch.mean(intensity_ir)
            dark_threshold = global_ir_mean
        elif dark_threshold_mode == 'custom':
            dark_threshold = custom_dark_threshold
        else:
            raise ValueError(f"Unsupported dark_threshold_mode: {dark_threshold_mode}")

        patch_darkness = torch.clamp((dark_threshold - patch_ir_max) / (dark_threshold + 1e-8), 0, 1)
        patch_weight = 1.0 - patch_darkness * 0.8

        adaptive_weights = pixel_weights * patch_weight
        target_intensity = adaptive_weights * intensity_ir + (1 - adaptive_weights) * intensity_vis

        return F.l1_loss(intensity_fusion, target_intensity)

    def _compute_gradient_loss(self, pred_vis, pred_ir, pred_fusion, targets):
        """梯度损失（调用内部的_gradient_loss方法）"""
        if self.lambda_dict['gradient'] == 0:
            return torch.tensor(0.0, device=self.device)

        target_vis = targets["img_vis"]
        target_ir = targets["img_ir"]
        target_fusion = targets.get("img_fusion", 0.5 * (target_vis + target_ir))

        # 替换原self.gradient_loss调用，直接用内部方法
        grad_loss_vis = self._gradient_loss(pred_vis, target_vis)
        grad_loss_ir = self._gradient_loss(pred_ir, target_ir)
        grad_loss_fusion = self._gradient_loss(pred_fusion, target_fusion)
        return grad_loss_vis + grad_loss_ir + grad_loss_fusion

    def _compute_perceptual_loss(self, pred_vis, pred_ir, pred_fusion, targets):
        """感知损失"""
        if self.lambda_dict['perceptual'] == 0 or self.vgg is None:
            return torch.tensor(0.0, device=self.device)

        target_vis = targets["img_vis"]
        target_ir = targets["img_ir"]
        target_fusion = targets.get("img_fusion", target_vis)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        def _single_perceptual(pred, target):
            if pred.shape[1] != 3:
                return torch.tensor(0.0, device=self.device)
            pred_norm = (pred - mean) / std
            target_norm = (target - mean) / std
            pred_feat = self.vgg(pred_norm)
            target_feat = self.vgg(target_norm)
            return F.l1_loss(pred_feat, target_feat)

        perceptual_vis = _single_perceptual(pred_vis, target_vis)
        perceptual_ir = _single_perceptual(pred_ir, target_ir)
        perceptual_fusion = _single_perceptual(pred_fusion, target_fusion)
        return perceptual_vis + perceptual_ir + perceptual_fusion

    def _compute_style_loss(self, pred_fusion, targets):
        """风格损失"""
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

    def _compute_pvs_loss(self, pred_fusion, targets):
        """PVS损失"""
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

    def _sobel_gradient(self, x):
        """Sobel梯度计算"""
        sobel_x = self.sobel_x.repeat(x.shape[1], 1, 1, 1)
        sobel_y = self.sobel_y.repeat(x.shape[1], 1, 1, 1)

        grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.shape[1])

        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    # ========== 前向传播（核心逻辑不变） ==========
    def forward(self, outputs, targets):
        pred_vis = outputs["img_vis_pred"]
        pred_ir = outputs["img_ir_pred"]
        pred_fusion = outputs["img_fusion_pred"]

        l1_vis = self._compute_l1_vis(pred_vis, targets["img_vis"])
        l1_ir = self._compute_l1_ir(pred_ir, targets["img_ir"])
        l1_fusion = self._compute_l1_fusion(pred_fusion, targets)
        grad_loss = self._compute_gradient_loss(pred_vis, pred_ir, pred_fusion, targets)
        perceptual_loss = self._compute_perceptual_loss(pred_vis, pred_ir, pred_fusion, targets)
        style_loss = self._compute_style_loss(pred_fusion, targets)
        pvs_loss = self._compute_pvs_loss(pred_fusion, targets)

        total_loss = (
                self.lambda_dict['vis'] * l1_vis +
                self.lambda_dict['ir'] * l1_ir +
                self.lambda_dict['fusion'] * l1_fusion +
                self.lambda_dict['gradient'] * grad_loss +
                self.lambda_dict['perceptual'] * perceptual_loss +
                self.lambda_dict['style'] * style_loss +
                self.lambda_dict['pvs'] * pvs_loss
        )

        return {
            "total_loss": total_loss,
            "l1_vis": l1_vis,
            "l1_ir": l1_ir,
            "l1_fusion": l1_fusion,
            "grad_loss": grad_loss,
            "perceptual_loss": perceptual_loss,
            "style_loss": style_loss,
            "content_loss": l1_fusion if "img_fusion" not in targets else torch.tensor(0.0, device=self.device),
            "pvs_loss": pvs_loss,
            "lambda_config": self.lambda_dict.copy()
        }