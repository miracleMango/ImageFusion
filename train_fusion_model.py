import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image
import logging
from tqdm import tqdm
import warnings
import multiprocessing
import numpy as np
# ======================== 新增：混合精度训练依赖 ========================
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
# ======================== 新增：TensorBoard相关依赖 ========================
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from datetime import datetime

from models.model import ImageFusionNetworkWithSourcePE
# ======================== 新增：导入自定义Loss类 ========================
from models.loss import Loss  # 确保loss.py和训练代码同目录，或替换为你的Loss类路径

warnings.filterwarnings('ignore')

# ======================== 全局配置（新增显著性目录） ========================
# 数据相关
DATA_ROOT = "./datasets/M3FD_Fusion_Patches_3900_128"
SALIENCY_ROOT = "./datasets/M3FD_Fusion_Saliency_Patches_3900_128"  # 新增：显著性图根目录
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
IMG_SUFFIX = [".png", ".jpg", ".jpeg", ".bmp"]

# 模型相关
VIS_IMG_CHANNELS = 3
IR_IMG_CHANNELS = 1
FEATURE_CHANNELS = 64
NUM_HEADS = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("当前使用的设备：", DEVICE)

# 训练相关
EPOCHS = 50
INIT_LR = 1e-4
OPTIMIZER_TYPE = "AdamW"
WEIGHT_DECAY = 1e-5
SCHEDULER_T_MAX = 40
SCHEDULER_ETA_MIN = 1e-6
# 梯度累积配置
USE_GRAD_ACCUM = False
GRAD_ACCUM_STEPS = 4 if USE_GRAD_ACCUM else 1
USE_MIXED_PRECISION = True if DEVICE == "cuda" else False

# 保存与日志/TensorBoard配置
SAVE_DIR = "./saved_models"
LOG_FILE = "./train.log"
SAVE_FREQ = 5
RESUME_TRAIN = False
RESUME_PATH = "./saved_models/latest_epoch_20.pth"
# TensorBoard配置
TB_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
TB_LOG_DIR = f"./runs/fusion_train_{TB_TIMESTAMP}"

# ======================== 数据范围调试函数（无修改） ========================
def debug_data_range(dataloader, num_batches=3):
    """调试数据范围，仅终端输出"""
    print("\n" + "=" * 60)
    print("数据范围调试信息")
    print("=" * 60)

    ir_stats = {"min": [], "max": [], "mean": [], "std": []}
    vis_stats = {"min": [], "max": [], "mean": [], "std": []}

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        img_ir = batch["ir"]
        img_vis = batch["vis"]

        # IR图像统计
        ir_stats["min"].append(img_ir.min().item())
        ir_stats["max"].append(img_ir.max().item())
        ir_stats["mean"].append(img_ir.mean().item())
        ir_stats["std"].append(img_ir.std().item())

        # VIS图像统计
        vis_stats["min"].append(img_vis.min().item())
        vis_stats["max"].append(img_vis.max().item())
        vis_stats["mean"].append(img_vis.mean().item())
        vis_stats["std"].append(img_vis.std().item())

        print(f"Batch {batch_idx + 1}:")
        print(f"  IR  - 范围: [{img_ir.min().item():.3f}, {img_ir.max().item():.3f}], "
              f"均值: {img_ir.mean().item():.3f}, 标准差: {img_ir.std().item():.3f}")
        print(f"  VIS - 范围: [{img_vis.min().item():.3f}, {img_vis.max().item():.3f}], "
              f"均值: {img_vis.mean().item():.3f}, 标准差: {img_vis.std().item():.3f}")

    # 汇总统计
    print("\n汇总统计:")
    print(f"IR图像:")
    print(f"  最小值范围: [{min(ir_stats['min']):.3f}, {max(ir_stats['min']):.3f}]")
    print(f"  最大值范围: [{min(ir_stats['max']):.3f}, {max(ir_stats['max']):.3f}]")
    print(f"  均值范围: [{min(ir_stats['mean']):.3f}, {max(ir_stats['mean']):.3f}]")
    print(f"  标准差范围: [{min(ir_stats['std']):.3f}, {max(ir_stats['std']):.3f}]")

    print(f"VIS图像:")
    print(f"  最小值范围: [{min(vis_stats['min']):.3f}, {max(vis_stats['min']):.3f}]")
    print(f"  最大值范围: [{min(vis_stats['max']):.3f}, {max(vis_stats['max']):.3f}]")
    print(f"  均值范围: [{min(vis_stats['mean']):.3f}, {max(vis_stats['mean']):.3f}]")
    print(f"  标准差范围: [{min(vis_stats['std']):.3f}, {max(vis_stats['std']):.3f}]")

    # 判断数据范围类型并推荐激活函数
    ir_min_global = min(ir_stats['min'])
    ir_max_global = max(ir_stats['max'])
    vis_min_global = min(vis_stats['min'])
    vis_max_global = max(vis_stats['max'])

    print(f"\n激活函数推荐:")
    if ir_min_global >= -0.1 and ir_max_global <= 1.1 and vis_min_global >= -0.1 and vis_max_global <= 1.1:
        print("  ✅ 数据在[0,1]范围内，推荐使用 Sigmoid 激活函数")
        print("  ❌ 数据在[0,1]范围内，不推荐使用 Tanh 激活函数")
    elif ir_min_global >= -1.1 and ir_max_global <= 1.1 and vis_min_global >= -1.1 and vis_max_global <= 1.1:
        print("  ✅ 数据在[-1,1]范围内，推荐使用 Tanh 激活函数")
        print("  ❌ 数据在[-1,1]范围内，不推荐使用 Sigmoid 激活函数")
    else:
        print("  ⚠️  数据范围异常，建议检查数据预处理")
        print(f"  IR范围: [{ir_min_global:.3f}, {ir_max_global:.3f}]")
        print(f"  VIS范围: [{vis_min_global:.3f}, {vis_max_global:.3f}]")

    print("=" * 60 + "\n")


def debug_model_outputs(model, dataloader, loss_module, device):
    """调试模型输出范围"""
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 2:
                break
                
            img_ir = batch["ir"].to(device)
            img_vis = batch["vis"].to(device)
            saliency_identifiers = batch["filename"]  # 获取所有标识
            
            outputs = model(img_ir, img_vis)
            targets = {"img_vis": img_vis, "img_ir": img_ir}
            
            # 传递显著性标识列表
            losses = loss_module.forward(outputs, targets, saliency_identifiers)
            
            print(f"Batch {batch_idx + 1}:")
            print(f"Batch {batch_idx + 1}:")
            print(f"  输入IR范围: [{img_ir.min().item():.3f}, {img_ir.max().item():.3f}]")
            print(f"  输入VIS范围: [{img_vis.min().item():.3f}, {img_vis.max().item():.3f}]")
            print(
                f"  重建IR范围: [{outputs['img_ir_pred'].min().item():.3f}, {outputs['img_ir_pred'].max().item():.3f}]")
            print(
                f"  重建VIS范围: [{outputs['img_vis_pred'].min().item():.3f}, {outputs['img_vis_pred'].max().item():.3f}]")
            print(
                f"  融合图像范围: [{outputs['img_fusion_pred'].min().item():.3f}, {outputs['img_fusion_pred'].max().item():.3f}]")

            # 检查亮度问题
            ir_brightness_diff = outputs['img_ir_pred'].mean() - img_ir.mean()
            vis_brightness_diff = outputs['img_vis_pred'].mean() - img_vis.mean()

            print(f"  亮度差异 - IR: {ir_brightness_diff.item():.3f}, VIS: {vis_brightness_diff.item():.3f}")
            print(f"  当前批次总损失: {losses['total_loss'].item():.4f}")

            if abs(ir_brightness_diff) > 0.1 or abs(vis_brightness_diff) > 0.1:
                print("  ⚠️  检测到亮度偏高问题!")

    print("=" * 60 + "\n")
    model.train()


# ======================== 红外/可见光Transform（无修改） ========================
def get_ir_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def get_vis_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


# ======================== Dataset（无修改） ========================
class FusionDataset(Dataset):
    def __init__(self, ir_dir, vis_dir, img_size, ir_transform=None, vis_transform=None):
        self.ir_dir = ir_dir
        self.vis_dir = vis_dir
        self.img_size = img_size
        self.ir_transform = ir_transform
        self.vis_transform = vis_transform

        self.ir_filenames = [f for f in os.listdir(ir_dir) if os.path.splitext(f)[-1].lower() in IMG_SUFFIX]
        self.vis_filenames = [f for f in os.listdir(vis_dir) if os.path.splitext(f)[-1].lower() in IMG_SUFFIX]

        self.common_filenames = list(set([os.path.splitext(f)[0] for f in self.ir_filenames]) &
                                     set([os.path.splitext(f)[0] for f in self.vis_filenames]))
        self.common_filenames.sort()

        print(f"找到 {len(self.common_filenames)} 对有效图像")
        if len(self.common_filenames) == 0:
            raise ValueError("未找到成对的红外/可见光图像！请检查文件名是否一致")

    def __len__(self):
        return len(self.common_filenames)

    def __getitem__(self, idx):
        base_name = self.common_filenames[idx]
        ir_path = None
        vis_path = None
        for suffix in IMG_SUFFIX:
            if os.path.exists(os.path.join(self.ir_dir, base_name + suffix)):
                ir_path = os.path.join(self.ir_dir, base_name + suffix)
            if os.path.exists(os.path.join(self.vis_dir, base_name + suffix)):
                vis_path = os.path.join(self.vis_dir, base_name + suffix)

        ir_img = Image.open(ir_path).convert("L")
        vis_img = Image.open(vis_path).convert("RGB")

        if self.ir_transform:
            ir_img = self.ir_transform(ir_img)
        if self.vis_transform:
            vis_img = self.vis_transform(vis_img)

        return {"ir": ir_img, "vis": vis_img, "filename": base_name}


# ======================== DataLoader（无修改） ========================
def get_dataloader():
    ir_transform = get_ir_transform()
    vis_transform = get_vis_transform()

    ir_dir = os.path.join(DATA_ROOT, "Ir")
    vis_dir = os.path.join(DATA_ROOT, "Vis")

    dataset = FusionDataset(ir_dir, vis_dir, IMG_SIZE, ir_transform, vis_transform)

    num_workers = 0 if os.name == 'nt' else multiprocessing.cpu_count() // 2
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True
    )
    return dataloader


# ======================== 日志配置（简化：仅终端输出关键信息） ========================
def init_logger():
    """简化日志配置：仅输出到终端，不写详细loss到文件"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]  # 仅控制台输出
    )
    logger = logging.getLogger(__name__)
    # 屏蔽不必要的第三方日志
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    return logger


# ======================== TensorBoard初始化 ========================
def init_tb_writer():
    """初始化TensorBoard Writer，输出启动命令"""
    os.makedirs(TB_LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=TB_LOG_DIR)
    print(f"\n✅ TensorBoard日志目录：{os.path.abspath(TB_LOG_DIR)}")
    print(f"✅ 启动命令：tensorboard --logdir={os.path.abspath(TB_LOG_DIR)} --port=6006")
    return writer


# ======================== 训练类（核心修改：适配新Loss类） ========================
class FusionTrainer:
    def __init__(self, model, dataloader, logger):
        self.model = model.to(DEVICE)
        self.dataloader = dataloader
        self.logger = logger

        # ======================== 新增：初始化自定义Loss模块 ========================
        self.loss_module = Loss(
            device=DEVICE,
            saliency_root=SALIENCY_ROOT,  # 预计算显著性图根目录
            grad_loss_type='l1',
            grad_reduction='mean'
        )
        self.logger.info(f"✅ 自定义Loss模块初始化完成，显著性目录：{SALIENCY_ROOT}")

        # 梯度累积配置
        self.use_grad_accum = USE_GRAD_ACCUM
        self.grad_accum_steps = GRAD_ACCUM_STEPS

        # 优化器
        if OPTIMIZER_TYPE == "AdamW":
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=INIT_LR,
                weight_decay=WEIGHT_DECAY
            )
        elif OPTIMIZER_TYPE == "Adam":
            self.optimizer = optim.Adam(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
        else:
            self.optimizer = optim.SGD(model.parameters(), lr=INIT_LR, momentum=0.9)

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=SCHEDULER_T_MAX,
            eta_min=SCHEDULER_ETA_MIN,
            last_epoch=-1
        )

        self.start_epoch = 1
        self.best_loss = float("inf")

        # 断点续训
        if RESUME_TRAIN and os.path.exists(RESUME_PATH):
            self._load_checkpoint(RESUME_PATH)
            self.logger.info(f"断点续训：加载 {RESUME_PATH}，从第 {self.start_epoch} 轮开始")

        os.makedirs(SAVE_DIR, exist_ok=True)

        # 加速组件
        self.scaler = GradScaler() if USE_MIXED_PRECISION else None
        if DEVICE == "cuda":
            cudnn.benchmark = True
            cudnn.deterministic = False
            self.logger.info("✅ 已开启cudnn benchmark加速")
        self.logger.info(f"✅ 混合精度训练: {USE_MIXED_PRECISION}")
        self.logger.info(
            f"✅ 梯度累积: {self.use_grad_accum} (等效batch size: {BATCH_SIZE * self.grad_accum_steps})")

        # ======================== 初始化TensorBoard ========================
        self.tb_writer = init_tb_writer()
        # 可视化模型结构（跳过，避免字典输出报错）
        self.logger.info("✅ 跳过模型结构图可视化（避免字典输出trace报错）")

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_loss = checkpoint["best_loss"]
        if USE_MIXED_PRECISION and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            self.logger.info("✅ 已加载混合精度scaler状态")

    def _save_checkpoint(self, epoch, loss):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "current_loss": loss
        }
        if USE_MIXED_PRECISION:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        latest_path = os.path.join(SAVE_DIR, f"latest_epoch_{epoch}.pth")
        torch.save(checkpoint, latest_path)
        self.logger.info(f"✅ 保存最新模型到：{latest_path}")

        if loss < self.best_loss:
            self.best_loss = loss
            best_path = os.path.join(SAVE_DIR, f"best_model_loss_{loss:.4f}.pth")
            torch.save(checkpoint["model_state_dict"], best_path)
            self.logger.info(f"✅ 保存最优模型到：{best_path}（损失：{loss:.4f}）")

        if epoch % SAVE_FREQ == 0:
            milestone_path = os.path.join(SAVE_DIR, f"milestone_epoch_{epoch}.pth")
            torch.save(checkpoint, milestone_path)
            self.logger.info(f"✅ 保存里程碑模型到：{milestone_path}")

    def _save_tb_image_samples(self, img_ir, img_vis, outputs, epoch):
        """保存图像样本到TensorBoard"""
        # 归一化到[0,1]
        ir_img = (img_ir[0].cpu() + 1) / 2
        vis_img = (img_vis[0].cpu() + 1) / 2
        fusion_img = (outputs['img_fusion_pred'][0].cpu() + 1) / 2

        # 单通道转3通道（方便显示）
        ir_img_3c = torch.cat([ir_img, ir_img, ir_img], dim=0) if ir_img.shape[0] == 1 else ir_img
        fusion_img_3c = torch.cat([fusion_img, fusion_img, fusion_img], dim=0) if fusion_img.shape[0] == 1 else fusion_img

        # 拼接成网格
        img_grid = vutils.make_grid(
            [ir_img_3c, vis_img, fusion_img_3c],
            nrow=3,
            normalize=False,
            scale_each=True
        )
        self.tb_writer.add_image("Sample/IR_VIS_Fusion", img_grid, epoch)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        # 新增：累计各细分损失
        loss_metrics = {
            "l1_vis": 0.0, "l1_ir": 0.0, "grad_loss": 0.0,
            "perceptual_loss": 0.0, "style_loss": 0.0, "pvs_loss": 0.0,
            "gradloss": 0.0, "intloss": 0.0, "maxintloss": 0.0,
            "color_loss": 0.0
        }

        pbar = tqdm(self.dataloader, desc=f"Epoch [{epoch}/{EPOCHS}]")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            img_ir = batch["ir"].to(DEVICE, non_blocking=True)
            img_vis = batch["vis"].to(DEVICE, non_blocking=True)
            # 关键：获取显著性标识（每个batch的第一个样本标识，或批量处理）
            saliency_identifiers = batch["filename"]  # 使用整个batch的标识列表

            # 混合精度前向传播
            with autocast(enabled=USE_MIXED_PRECISION):
                outputs = self.model(img_ir, img_vis)
                targets = {"img_vis": img_vis, "img_ir": img_ir}
                # 核心修改：调用自定义Loss计算损失，传递显著性标识
                losses = self.loss_module.forward(outputs, targets, saliency_identifiers)

                # 梯度累积损失调整
                loss = losses["total_loss"] / self.grad_accum_steps if self.use_grad_accum else losses["total_loss"]

            # 反向传播
            if USE_MIXED_PRECISION:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 参数更新
            if self.use_grad_accum:
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if USE_MIXED_PRECISION:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
            else:
                if USE_MIXED_PRECISION:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # 每10轮保存一次图像样本到TensorBoard
            if epoch % 10 == 0 and batch_idx == 0:
                self._save_tb_image_samples(img_ir, img_vis, outputs, epoch)

            # 累计损失
            total_loss += losses["total_loss"].item()

            # 进度条仅展示核心指标（简化输出）
            pbar.set_postfix({
                "total_loss": f"{losses['total_loss'].item():.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                "grad_accum": f"{self.use_grad_accum}"
            })

        # 计算epoch平均损失
        avg_loss = total_loss / len(self.dataloader)
        avg_metrics = {k: v / len(self.dataloader) for k, v in loss_metrics.items()}
        current_lr = self.optimizer.param_groups[0]['lr']

        # 写入TensorBoard（使用平均损失）
        self.tb_writer.add_scalar("Loss/Total_Loss", avg_loss, epoch)
        for k, v in avg_metrics.items():
            self.tb_writer.add_scalar(f"Loss/{k}", v, epoch)
        self.tb_writer.add_scalar("Optimizer/Learning_Rate", current_lr, epoch)
        self.scheduler.step()
        return avg_loss

    def train(self):
        self.logger.info("=" * 50)
        self.logger.info(f"训练开始！设备：{DEVICE}，批次大小：{BATCH_SIZE}，总轮次：{EPOCHS}")
        self.logger.info(f"模型参数：特征通道数={FEATURE_CHANNELS}，注意力头数={NUM_HEADS}")
        self.logger.info(f"显著性图目录：{os.path.abspath(SALIENCY_ROOT)}")
        self.logger.info(f"TensorBoard日志路径：{os.path.abspath(TB_LOG_DIR)}")
        self.logger.info("=" * 50)

        for epoch in range(self.start_epoch, EPOCHS + 1):
            avg_loss = self.train_one_epoch(epoch)
            self._save_checkpoint(epoch, avg_loss)

        # 关闭TensorBoard Writer
        self.tb_writer.close()

        self.logger.info("=" * 50)
        self.logger.info(f"训练结束！最优损失：{self.best_loss:.4f}")
        self.logger.info(f"TensorBoard可视化命令：tensorboard --logdir={os.path.abspath(TB_LOG_DIR)} --port=6006")
        self.logger.info("=" * 50)


# ======================== 入口函数（修改调试函数调用） ========================
if __name__ == "__main__":
    # 初始化日志
    logger = init_logger()
    logger.info("加载数据集...")
    dataloader = get_dataloader()

    # 验证通道数
    for batch in dataloader:
        print("验证img_ir通道数：", batch["ir"].shape)
        print("验证img_vis通道数：", batch["vis"].shape)
        print("验证filename标识：", batch["filename"][0])
        break

    # 数据范围调试
    debug_data_range(dataloader)

    logger.info("初始化模型...")
    model = ImageFusionNetworkWithSourcePE(
        vis_img_channels=VIS_IMG_CHANNELS,
        ir_img_channels=IR_IMG_CHANNELS,
        feature_channels=FEATURE_CHANNELS,
        num_heads=NUM_HEADS,
        use_position_encoding=True
    )

    model = model.to(DEVICE)
    print(f"模型已移动到设备: {DEVICE}")

    # 初始化Loss模块用于调试
    loss_module = Loss(
        device=DEVICE,
        saliency_root=SALIENCY_ROOT,
    )

    # 模型输出调试（传入Loss模块）
    debug_model_outputs(model, dataloader, loss_module, DEVICE)

    # 启动训练
    trainer = FusionTrainer(model, dataloader, logger)
    trainer.train()
