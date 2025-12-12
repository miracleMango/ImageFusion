import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Loss(nn.Module):  # 仅保留gradloss和intloss的损失类
    def __init__(self,
                 device,
                 # 基础损失权重
                 lambda_vis=1.0,
                 lambda_ir=1.0,
                 lambda_perceptual=0,
                 lambda_gradient=0,
                 lambda_style=0,
                 lambda_pvs=0.1,
                 # 拆分后的gradloss/intloss独立权重
                 lambda_gradloss=1.0,
                 lambda_intloss=0.01,
                 # GradientLoss相关参数
                 grad_loss_type='l1',
                 grad_reduction='mean'):
        super().__init__()
        self.device = device

        # ========== 损失权重配置 ==========
        self.lambda_dict = {
            'vis': lambda_vis,
            'ir': lambda_ir,
            'perceptual': lambda_perceptual,
            'gradient': lambda_gradient,
            'style': lambda_style,
            'pvs': lambda_pvs,
            'gradloss': lambda_gradloss,
            'intloss': lambda_intloss
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

    # ========== 原GradientLoss核心方法（不变） ==========
    def _compute_gradient(self, x, kernel):
        """计算单方向梯度（x/y），使用 Replicate Padding 消除边缘黑线"""
        b, c, h, w = x.shape
        kernel = kernel.repeat(c, 1, 1, 1)
        # 1. 计算需要的 padding 大小
        pad_size = (kernel.size(-1) - 1) // 2
        # 2. 关键修改：先手动进行 replicate padding
        # pad顺序: (左, 右, 上, 下)
        x_padded = F.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='replicate')

        # 3. 卷积时设 padding=0
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

    # ========== 合并后：_compute_gradloss（独立完整） ==========
    def _compute_gradloss(self, pred_fusion, target_vis, target_ir):
        """
        计算内容保持拆分后的梯度损失（gradloss）
        参数：
            pred_fusion: 预测的融合图像 [B, C, H, W]
            target_vis: 目标可见光图像 [B, C, H, W]
            target_ir: 目标红外图像 [B, C, H, W]
        返回：
            gradloss: 梯度匹配损失值
        """
        # 计算各图像的Sobel梯度（取绝对值）
        grad_fusion = torch.abs(self._sobel_gradient(pred_fusion))
        grad_vis = torch.abs(self._sobel_gradient(target_vis))
        grad_ir = torch.abs(self._sobel_gradient(target_ir))
        # 融合图像梯度需匹配可见光/红外梯度的最大值
        gradloss = F.mse_loss(grad_fusion, torch.max(grad_vis, grad_ir))
        return gradloss

    # ========== 合并后：_compute_intloss（包含所有强度损失逻辑，无冗余调用） ==========
    def _compute_intloss(self, pred_fusion, target_vis, target_ir):
        """
        计算内容保持拆分后的强度损失（intloss）
        参数：
            pred_fusion: 预测的融合图像 [B, C, H, W]
            target_vis: 目标可见光图像 [B, C, H, W]
            target_ir: 目标红外图像 [B, C, H, W]
        返回：
            intloss: 强度匹配损失值
        """
        # 1. 确保红外图像维度与可见光/融合图像一致（单通道→3通道）
        intensity_fusion = pred_fusion
        intensity_vis = target_vis
        intensity_ir = target_ir.repeat(1, 3, 1, 1)

        # 2. 计算像素级max(可见光, 红外)
        max_vis_ir = torch.max(intensity_vis, intensity_ir)
        # 3. 融合图像与max值的像素差异（取绝对值，避免正负抵消）
        pixel_diff = torch.abs(intensity_fusion - max_vis_ir)

        # 4. 归一化损失（除以图像像素总数，标准化损失值）
        B, C, H, W = pixel_diff.shape
        intloss = pixel_diff.sum() / (H * W)
        return intloss

    # ========== 全局梯度损失（不变） ==========
    def _compute_gradient_loss(self, pred_vis, pred_ir, pred_fusion, targets):
        """计算全局梯度损失（与gradloss无关，是预测图vs目标图的梯度损失）"""
        if self.lambda_dict['gradient'] == 0:
            return torch.tensor(0.0, device=self.device)

        target_vis = targets["img_vis"]
        target_ir = targets["img_ir"]
        target_fusion = 0.5 * (target_vis + target_ir)

        grad_loss_vis = self._gradient_loss(pred_vis, target_vis)
        grad_loss_ir = self._gradient_loss(pred_ir, target_ir)
        grad_loss_fusion = self._gradient_loss(pred_fusion, target_fusion)
        return grad_loss_vis + grad_loss_ir + grad_loss_fusion

    # ========== 感知损失（不变） ==========
    def _compute_perceptual_loss(self, pred_vis, pred_ir, pred_fusion, targets):
        """计算感知损失"""
        if self.lambda_dict['perceptual'] == 0 or self.vgg is None:
            return torch.tensor(0.0, device=self.device)

        target_vis = targets["img_vis"]
        target_ir = targets["img_ir"]
        target_fusion = target_vis

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

        # 1. 关键修改：手动 Replicate Padding (Sobel核通常是3x3，所以pad=1)
        x_padded = F.pad(x, (1, 1, 1, 1), mode='replicate')

        # 2. 卷积时 padding=0
        grad_x = F.conv2d(x_padded, sobel_x, padding=0, groups=x.shape[1])
        grad_y = F.conv2d(x_padded, sobel_y, padding=0, groups=x.shape[1])

        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    # ========== 前向传播（调用精简后的独立方法） ==========
    def forward(self, outputs, targets):
        pred_vis = outputs["img_vis_pred"]
        pred_ir = outputs["img_ir_pred"]
        pred_fusion = outputs["img_fusion_pred"]
        target_vis = targets["img_vis"]
        target_ir = targets["img_ir"]

        # 1. 计算基础L1损失
        l1_vis = self._compute_l1_vis(pred_vis, target_vis)
        l1_ir = self._compute_l1_ir(pred_ir, target_ir)

        # 2. 调用精简后的方法计算gradloss/intloss
        gradloss = self._compute_gradloss(pred_fusion, target_vis, target_ir)
        intloss = self._compute_intloss(pred_fusion, target_vis, target_ir)

        # 3. 计算其他损失项
        grad_loss = self._compute_gradient_loss(pred_vis, pred_ir, pred_fusion, targets)
        perceptual_loss = self._compute_perceptual_loss(pred_vis, pred_ir, pred_fusion, targets)
        style_loss = self._compute_style_loss(pred_fusion, targets)
        pvs_loss = self._compute_pvs_loss(pred_fusion, targets)

        # 4. 总损失计算（权重加权求和）
        total_loss = (
                self.lambda_dict['vis'] * l1_vis +
                self.lambda_dict['ir'] * l1_ir +
                self.lambda_dict['gradloss'] * gradloss +
                self.lambda_dict['intloss'] * intloss +
                self.lambda_dict['gradient'] * grad_loss +
                self.lambda_dict['perceptual'] * perceptual_loss +
                self.lambda_dict['style'] * style_loss +
                self.lambda_dict['pvs'] * pvs_loss
        )

        # 5. 返回字典（无冗余字段）
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
            "lambda_config": self.lambda_dict.copy()
        }