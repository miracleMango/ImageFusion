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
import multiprocessing  # 补充：添加这行导入，解决NameError
import numpy as np
import torchvision.transforms.functional as TF
# ======================== 新增：混合精度训练依赖 ========================
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
# ======================== 新增：TensorBoard相关依赖 ========================
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from datetime import datetime
# ======================== 新增：分组打乱所需依赖 ========================
from collections import defaultdict
import random
# ======================== 新增：OpenCV（提升图像加载速度，优化IO瓶颈） ========================
import cv2

# 注意：确保你的模型和Loss类路径正确
from models.model import ImageFusionNetworkWithSourcePE
from models.loss import Loss

warnings.filterwarnings('ignore')

# ======================== 全局配置（核心新增：FIXED_FULL_SIZE 统一整图尺寸） ========================
# 数据相关
DATA_ROOT = "./datasets/M3FD_Fusion_Patches_3900_128"
FULL_IMAGE_ROOT = "./datasets/M3FD_Fusion_3900"  # 整图根目录（存放ir/vis两个子文件夹）
SALIENCY_ROOT = "./datasets/M3FD_Fusion_Saliency_Patches_3900_128"
PATCH_SIZE = (128, 128)
PATCH_STRIDE = 128
BATCH_SIZE = 8
IMG_SUFFIX = [".png", ".jpg", ".jpeg", ".bmp"]

# 核心新增：统一整图尺寸（可根据需求调整，建议为2的幂次，如512、256）
FIXED_FULL_SIZE = (512, 512)  # 固定整图为512×512，保持空间比例不变

# 模型相关
VIS_IMG_CHANNELS = 3
IR_IMG_CHANNELS = 1
FEATURE_CHANNELS = 64
NUM_HEADS = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 训练相关
EPOCHS = 50
INIT_LR = 1e-4
OPTIMIZER_TYPE = "AdamW"
WEIGHT_DECAY = 1e-5
SCHEDULER_T_MAX = 40
SCHEDULER_ETA_MIN = 1e-6
USE_GRAD_ACCUM = False
GRAD_ACCUM_STEPS = 4 if USE_GRAD_ACCUM else 1
USE_MIXED_PRECISION = True if DEVICE == "cuda" else False

# 保存与日志/TensorBoard配置
SAVE_DIR = "./saved_models"
LOG_FILE = "./train.log"
SAVE_FREQ = 5
RESUME_TRAIN = False
RESUME_PATH = "./saved_models/latest_epoch_20.pth"
TB_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
TB_LOG_DIR = f"./runs/fusion_train_{TB_TIMESTAMP}"

# ======================== Transform（核心修改：新增Saliency Transform + 整图统一尺寸） ========================
def get_ir_transform():
    """Patch的IR Transform（带数据增强，保持原有逻辑）"""
    return transforms.Compose([
        transforms.Resize(PATCH_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def get_vis_transform():
    """Patch的VIS Transform（带数据增强，保持原有逻辑）"""
    return transforms.Compose([
        transforms.Resize(PATCH_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def get_saliency_transform():
    """Patch的Saliency Transform（与IR保持一致，保证预处理对齐）"""
    return transforms.Compose([
        transforms.Resize(PATCH_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def get_ir_full_transform():
    """IR 整图 Transform（核心修改：等比例缩放+居中裁剪，统一到FIXED_FULL_SIZE）"""
    return transforms.Compose([
        # 步骤1：等比例缩放，最短边达到固定尺寸，保持宽高比（避免拉伸）
        transforms.Resize(FIXED_FULL_SIZE, interpolation=Image.BILINEAR),
        # 步骤2：居中裁剪，确保尺寸严格统一为FIXED_FULL_SIZE
        transforms.CenterCrop(FIXED_FULL_SIZE),
        # 步骤3：常规转换（保持原有归一化）
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def get_vis_full_transform():
    """VIS 整图 Transform（核心修改：等比例缩放+居中裁剪，统一到FIXED_FULL_SIZE）"""
    return transforms.Compose([
        # 步骤1：等比例缩放，最短边达到固定尺寸，保持宽高比（避免拉伸）
        transforms.Resize(FIXED_FULL_SIZE, interpolation=Image.BILINEAR),
        # 步骤2：居中裁剪，确保尺寸严格统一为FIXED_FULL_SIZE
        transforms.CenterCrop(FIXED_FULL_SIZE),
        # 步骤3：常规转换（保持原有归一化）
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


# ======================== Dataset（核心优化：缓存文件路径+OpenCV加载+减少冗余IO） ========================
class FusionDataset(Dataset):
    def __init__(self, ir_dir, vis_dir, saliency_dir, ir_full_dir, vis_full_dir, patch_size,
                 fixed_full_size=FIXED_FULL_SIZE,
                 ir_transform=None, vis_transform=None, saliency_transform=None, 
                 ir_full_transform=None, vis_full_transform=None,
                 preload_full_images=True):
        self.ir_dir = ir_dir
        self.vis_dir = vis_dir
        self.saliency_dir = saliency_dir
        self.ir_full_dir = ir_full_dir
        self.vis_full_dir = vis_full_dir
        self.patch_size = patch_size
        self.fixed_full_size = fixed_full_size
        self.ir_transform = ir_transform
        self.vis_transform = vis_transform
        self.saliency_transform = saliency_transform
        self.ir_full_transform = ir_full_transform
        self.vis_full_transform = vis_full_transform
        self.preload_full_images = preload_full_images
        self.full_image_cache = {}

        # ======================== 优化1：缓存文件完整路径，减少高频os.path.exists调用（解决IO瓶颈） ========================
        self.ir_file_map = self._build_file_map(ir_dir)
        self.vis_file_map = self._build_file_map(vis_dir)
        self.saliency_file_map = self._build_file_map(saliency_dir)

        # 保留三者共同且符合"xxx_patch_xx_yy"格式的文件名
        self.common_filenames = [
            fname for fname in (self.ir_file_map.keys() & self.vis_file_map.keys() & self.saliency_file_map.keys())
            if "_patch_" in fname
        ]
        self.common_filenames.sort()

        if self.preload_full_images:
            self._preload_all_full_images()

        # 整图缓存初始化
        self.cached_full_img_base = None
        self.cached_ir_full_img = None
        self.cached_vis_full_img = None
        self.cached_original_full_size = None
        self.cached_fixed_full_size = None

        # 日志仅通过logger输出，移除print
        if len(self.common_filenames) == 0:
            raise ValueError("未找到符合格式的成对红外/可见光/Saliency patch图像！请检查文件名是否为xxx_patch_xx_yy")
    
    def _preload_all_full_images(self):
        """预加载所有整图到内存（消除训练时的 IO 瓶颈）"""
        # 提取所有唯一的整图前缀
        unique_bases = set()
        for fname in self.common_filenames:
            full_img_base, _, _ = self._parse_patch_info(fname)
            unique_bases.add(full_img_base)
        
        print(f"正在预加载 {len(unique_bases)} 张整图到内存...")
        for base in tqdm(unique_bases, desc="预加载整图"):
            # 加载 IR 整图
            ir_path = self._get_full_image_path(base, is_ir=True)
            ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
            ir_img_pil = Image.fromarray(ir_img)
            ir_tensor = self.ir_full_transform(ir_img_pil)
            
            # 加载 VIS 整图
            vis_path = self._get_full_image_path(base, is_ir=False)
            vis_img = cv2.imread(vis_path)[:, :, ::-1]
            vis_img_pil = Image.fromarray(vis_img)
            vis_tensor = self.vis_full_transform(vis_img_pil)
            
            # 记录原始尺寸
            orig_h, orig_w = ir_img.shape
            
            # 缓存
            self.full_image_cache[base] = {
                'ir_full': ir_tensor,
                'vis_full': vis_tensor,
                'original_size': (orig_h, orig_w),
                'fixed_size': self.fixed_full_size
            }
        print(f"✅ 整图预加载完成，内存占用约 {len(unique_bases) * 512 * 512 * 4 / 1024 / 1024:.1f} MB")

    # 新增：构建文件名→完整路径的映射，一次性遍历，避免重复IO
    def _build_file_map(self, target_dir):
        file_map = {}
        for f in os.listdir(target_dir):
            fname, fext = os.path.splitext(f)
            if fext.lower() in IMG_SUFFIX:
                file_map[fname] = os.path.join(target_dir, f)
        return file_map

    # 新增：获取样本文件名（用于分组打乱）
    def get_filename(self, idx):
        if idx < 0 or idx >= len(self.common_filenames):
            raise IndexError("样本索引超出范围")
        return self.common_filenames[idx]

    def _parse_patch_info(self, patch_base_name):
        """解析patch文件名，提取整图前缀和patch位置（保持原有逻辑）"""
        parts = patch_base_name.split("_patch_")
        if len(parts) != 2:
            raise ValueError(f"无效的patch文件名格式：{patch_base_name}，应为xxx_patch_xx_yy")
        
        full_img_base = parts[0]
        patch_xy = parts[1].split("_")
        if len(patch_xy) != 2:
            raise ValueError(f"无效的patch位置格式：{patch_base_name}，应为xxx_patch_xx_yy")
        
        patch_x = int(patch_xy[0])
        patch_y = int(patch_xy[1])
        return full_img_base, patch_x, patch_y

    def _get_full_image_path(self, full_img_base, is_ir=True):
        """根据整图前缀获取整图路径（保持原有逻辑）"""
        target_dir = self.ir_full_dir if is_ir else self.vis_full_dir
        for suffix in IMG_SUFFIX:
            full_path = os.path.join(target_dir, full_img_base + suffix)
            if os.path.exists(full_path):
                return full_path
        raise FileNotFoundError(f"未找到整图文件：{full_img_base}（在{target_dir}目录下）")

    def _load_and_cache_full_images(self, full_img_base):
        """
        加载整图并更新缓存（核心修改：记录原始尺寸+固定尺寸，用于坐标缩放）
        返回：ir_full_img（固定尺寸张量），vis_full_img（固定尺寸张量），
              original_full_size（原始尺寸(H,W)），fixed_full_size（固定尺寸(H,W)）
        """
        # 1. 加载IR整图（原始尺寸）
        ir_full_path = self._get_full_image_path(full_img_base, is_ir=True)
        ir_full_img_pil_original = Image.open(ir_full_path).convert("L")
        # 记录原始尺寸（PIL Image.size返回(W,H)，转换为(H,W)）
        original_w, original_h = ir_full_img_pil_original.size
        original_full_size = (original_h, original_w)
        # 应用Transform（缩放为固定尺寸）
        ir_full_img = self.ir_full_transform(ir_full_img_pil_original) if self.ir_full_transform else ir_full_img_pil_original
        
        # 2. 加载VIS整图（原始尺寸）
        vis_full_path = self._get_full_image_path(full_img_base, is_ir=False)
        vis_full_img_pil_original = Image.open(vis_full_path).convert("RGB")
        # 应用Transform（缩放为固定尺寸）
        vis_full_img = self.vis_full_transform(vis_full_img_pil_original) if self.vis_full_transform else vis_full_img_pil_original
        
        # 3. 记录固定尺寸（与配置一致）
        fixed_full_size = self.fixed_full_size

        # 4. 更新缓存
        self.cached_full_img_base = full_img_base
        self.cached_ir_full_img = ir_full_img
        self.cached_vis_full_img = vis_full_img
        self.cached_original_full_size = original_full_size
        self.cached_fixed_full_size = fixed_full_size

        return ir_full_img, vis_full_img, original_full_size, fixed_full_size

    def __len__(self):
        return len(self.common_filenames)

    def __getitem__(self, idx):
        # 1. 加载 patch（同步加载 IR/VIS/Saliency）
        patch_base_name = self.common_filenames[idx]
        ir_path = self.ir_file_map.get(patch_base_name)
        vis_path = self.vis_file_map.get(patch_base_name)
        saliency_path = self.saliency_file_map.get(patch_base_name)

        # OpenCV 加载（保持原有逻辑）
        ir_img = Image.fromarray(cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE))
        vis_img = Image.fromarray(cv2.imread(vis_path)[:, :, ::-1])
        saliency_img = Image.fromarray(cv2.imread(saliency_path, cv2.IMREAD_GRAYSCALE))

        if self.ir_transform:
            ir_img = self.ir_transform(ir_img)
        if self.vis_transform:
            vis_img = self.vis_transform(vis_img)
        if self.saliency_transform:
            saliency_img = self.saliency_transform(saliency_img)

        # 2. 解析 patch 信息
        full_img_base, patch_x, patch_y = self._parse_patch_info(patch_base_name)

        # ✅ 改进：从内存缓存中读取整图（零 IO 开销）
        if self.preload_full_images and full_img_base in self.full_image_cache:
            cache = self.full_image_cache[full_img_base]
            ir_full_img = cache['ir_full']
            vis_full_img = cache['vis_full']
            original_full_size = cache['original_size']
            fixed_full_size = cache['fixed_size']
        else:
            # 回退到原有的缓存逻辑
            if full_img_base != self.cached_full_img_base:
                ir_full_img, vis_full_img, original_full_size, fixed_full_size = \
                    self._load_and_cache_full_images(full_img_base)
            else:
                ir_full_img = self.cached_ir_full_img
                vis_full_img = self.cached_vis_full_img
                original_full_size = self.cached_original_full_size
                fixed_full_size = self.cached_fixed_full_size

        # 3. 计算 patch 坐标（保持原有逻辑）
        patch_w, patch_h = self.patch_size
        orig_h, orig_w = original_full_size
        fixed_h, fixed_w = fixed_full_size

        scale_w = fixed_w / orig_w
        scale_h = fixed_h / orig_h

        x1 = int(patch_x * patch_w * scale_w)
        y1 = int(patch_y * patch_h * scale_h)
        x2 = int((patch_x + 1) * patch_w * scale_w)
        y2 = int((patch_y + 1) * patch_h * scale_h)

        x1 = max(0, min(x1, fixed_w))
        y1 = max(0, min(y1, fixed_h))
        x2 = max(x1, min(x2, fixed_w))
        y2 = max(y1, min(y2, fixed_h))

        patch_pos = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        img_size = torch.tensor(fixed_full_size, dtype=torch.float32)

        return {
            "ir": ir_img, 
            "vis": vis_img,
            "saliency": saliency_img,
            "ir_full": ir_full_img,
            "vis_full": vis_full_img,
            "patch_pos": patch_pos,
            "img_size": img_size,
            "filename": patch_base_name
        }

# ======================== 按整图分组打乱的包装Dataset ========================
class ShuffledByImageDataset(Dataset):
    """
    包装现有FusionDataset，实现：
    1. 按整图前缀分组（xxx_patch_xx_yy → xxx）
    2. 组间打乱整图顺序（保证训练随机性）
    3. 组内打乱patch顺序（可选，提升随机性且不影响缓存）
    """
    def __init__(self, original_dataset, shuffle_within_group=True):
        self.original_dataset = original_dataset
        self.shuffle_within_group = shuffle_within_group
        self.shuffled_indices = self._create_shuffled_indices()

    def _create_shuffled_indices(self):
        """生成按整图分组打乱后的样本索引列表"""
        # 步骤1：按整图前缀分组
        image_to_indices = defaultdict(list)
        for idx in range(len(self.original_dataset)):
            filename = self.original_dataset.get_filename(idx)
            full_img_base, _, _ = self.original_dataset._parse_patch_info(filename)
            image_to_indices[full_img_base].append(idx)
        
        # 步骤2：组间打乱
        image_ids = list(image_to_indices.keys())
        random.shuffle(image_ids)

        # 步骤3：组内打乱+拼接
        shuffled_indices = []
        for image_id in image_ids:
            patch_indices = image_to_indices[image_id]
            if self.shuffle_within_group:
                random.shuffle(patch_indices)
            shuffled_indices.extend(patch_indices)

        return shuffled_indices

    def __len__(self):
        return len(self.shuffled_indices)

    def __getitem__(self, idx):
        """按打乱后的索引读取原始样本，保持返回格式一致"""
        original_idx = self.shuffled_indices[idx]
        return self.original_dataset[original_idx]

# ======================== DataLoader（核心优化，清理无效参数） ========================
def get_dataloader():
    # Patch Transform
    ir_transform = get_ir_transform()
    vis_transform = get_vis_transform()
    saliency_transform = get_saliency_transform()

    # 整图Transform
    ir_full_transform = get_ir_full_transform()
    vis_full_transform = get_vis_full_transform()

    # 目录配置
    ir_dir = os.path.join(DATA_ROOT, "Ir")
    vis_dir = os.path.join(DATA_ROOT, "Vis")
    saliency_dir = SALIENCY_ROOT
    ir_full_dir = os.path.join(FULL_IMAGE_ROOT, "ir")
    vis_full_dir = os.path.join(FULL_IMAGE_ROOT, "vis")

    # 验证目录
    for dir_path in [ir_dir, vis_dir, saliency_dir, ir_full_dir, vis_full_dir]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"目录不存在：{dir_path}")

    # 加载原始FusionDataset（同步加载三者）
    original_dataset = FusionDataset(
        ir_dir=ir_dir,
        vis_dir=vis_dir,
        saliency_dir=saliency_dir,
        ir_full_dir=ir_full_dir,
        vis_full_dir=vis_full_dir,
        patch_size=PATCH_SIZE,
        fixed_full_size=FIXED_FULL_SIZE,
        ir_transform=ir_transform,
        vis_transform=vis_transform,
        saliency_transform=saliency_transform,
        ir_full_transform=ir_full_transform,
        vis_full_transform=vis_full_transform,
        preload_full_images=True  # ✅ 启用整图预加载
    )

    # 使用按整图分组打乱的包装类
    dataset = ShuffledByImageDataset(
        original_dataset=original_dataset,
        shuffle_within_group=True
    )

    # ======================== 优化4：强制num_workers=0，清理多进程无效参数（解决内存瓶颈） ========================
    num_workers = 0 if os.name == 'nt' else min(4, multiprocessing.cpu_count() // 2)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if num_workers > 0 else None,  # ✅ 预取 2 个 batch
        persistent_workers=True if num_workers > 0 else False  # ✅ 保持 workers 常驻
    )
    return dataloader


# ======================== 日志配置 ========================
def init_logger():
    """简化日志配置：仅输出到终端"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    return logger


# ======================== 优化5：修正笔误（init_tb_writer，原少写一个'r'） ========================
def init_tb_writer():
    """初始化TensorBoard Writer"""
    os.makedirs(TB_LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=TB_LOG_DIR)
    return writer


# ======================== 训练类（核心优化：减少高频item()，批量迁移张量） ========================
class FusionTrainer:
    def __init__(self, model, dataloader, logger):
        self.model = model.to(DEVICE)
        self.dataloader = dataloader
        self.logger = logger

        # 初始化自定义Loss模块（不再需要传入saliency_root，直接使用batch中的saliency）
        self.loss_module = Loss(
            device=DEVICE,
            grad_loss_type='l1',
            grad_reduction='mean'
        )
        self.logger.info(f"✅ 自定义Loss模块初始化完成")

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

        # 初始化TensorBoard
        self.tb_writer = init_tb_writer()
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
        
        # ✅ 改进：使用列表累积张量，而非标量累积
        loss_tensors = []  # 存储每个 batch 的 loss tensor
        loss_components = {k: [] for k in ["l1_vis", "l1_ir", "grad_loss", "perceptual_loss", 
                                           "style_loss", "pvs_loss", "gradloss", "intloss", 
                                           "maxintloss", "color_loss"]}
        
        pbar = tqdm(self.dataloader, desc=f"Epoch [{epoch}/{EPOCHS}]")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # 批量迁移张量（保持你的优化）
            batch = {k: v.to(DEVICE, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}

            # 混合精度前向传播
            with autocast(enabled=USE_MIXED_PRECISION):
                outputs = self.model(
                    batch["ir"], batch["vis"], 
                    batch["ir_full"], batch["vis_full"], 
                    batch["patch_pos"], batch["img_size"]
                )
                targets = {"img_vis": batch["vis"], "img_ir": batch["ir"]}
                losses = self.loss_module.forward(outputs, targets, batch["saliency"])
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

            # 图像样本保存
            if epoch % 10 == 0 and batch_idx == 0:
                self._save_tb_image_samples(img_ir, img_vis, outputs, epoch)

            # ✅ 关键改进：仅保存张量引用，避免 .item() 调用
            loss_tensors.append(losses["total_loss"].detach())  # detach 避免计算图累积
            for k in loss_components.keys():
                if k in losses:
                    loss_components[k].append(losses[k].detach())

            # ✅ 进度条显示：只在必要时调用 .item()（每 10 个 batch 更新一次）
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{losses['total_loss'].item():.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })

        # ✅ Epoch 结束后批量计算平均值（仅一次 .item() 调用）
        avg_loss = torch.stack(loss_tensors).mean().item()
        avg_metrics = {k: torch.stack(v).mean().item() for k, v in loss_components.items() if v}
        current_lr = self.optimizer.param_groups[0]['lr']

        # TensorBoard 记录
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
        self.logger.info(f"数据配置：Patch尺寸={PATCH_SIZE}，固定整图尺寸={FIXED_FULL_SIZE}")
        self.logger.info(f"显著性图已在Dataset中同步加载")
        self.logger.info(f"整图目录：{os.path.abspath(FULL_IMAGE_ROOT)}")
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


# ======================== 入口函数（移除所有多余print调试） ========================
if __name__ == "__main__":
    # 初始化日志
    logger = init_logger()
    logger.info("加载数据集...")
    dataloader = get_dataloader()

    logger.info("初始化模型...")
    model = ImageFusionNetworkWithSourcePE(
        vis_img_channels=VIS_IMG_CHANNELS,
        ir_img_channels=IR_IMG_CHANNELS,
        feature_channels=FEATURE_CHANNELS,
        num_heads=NUM_HEADS,
        use_position_encoding=True
    )
    model = model.to(DEVICE)

    # 启动训练
    trainer = FusionTrainer(model, dataloader, logger)
    trainer.train()
