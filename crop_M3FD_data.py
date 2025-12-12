import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# ===================== 配置项（与训练代码对齐）=====================
# 原数据集路径（和训练代码一致）
ORIGINAL_DATA_ROOT = "./datasets/M3FD_Fusion_3900"
# 新生成Patch的数据集路径（保持同级目录，结构和原数据集一致）
PATCH_DATA_ROOT = "./datasets/M3FD_Fusion_Patches_3900_128"
# Patch尺寸（和训练代码的IMG_SIZE对齐）
PATCH_SIZE = (128, 128)  # (高, 宽)
# 支持的图像后缀（和训练代码一致）
IMG_SUFFIX = [".png", ".jpg", ".jpeg", ".bmp"]
# 边界处理方式："zero"补零（保证所有Patch都是256x256），"trunc"截断（仅保留有效区域）
PADDING_MODE = "zero"
# 是否保存单通道红外图像（True：保存为灰度图，False：转为3通道）
SAVE_IR_AS_GRAY = True


# ===================== 核心工具函数 =====================
def create_dir_structure():
    """创建和原数据集一致的目录结构（Ir/Vis子文件夹）"""
    ir_patch_dir = os.path.join(PATCH_DATA_ROOT, "Ir")
    vis_patch_dir = os.path.join(PATCH_DATA_ROOT, "Vis")
    os.makedirs(ir_patch_dir, exist_ok=True)
    os.makedirs(vis_patch_dir, exist_ok=True)
    return ir_patch_dir, vis_patch_dir


def get_paired_filenames():
    """获取原数据集中成对的IR/VIS文件名（和训练代码逻辑完全一致）"""
    # 原数据集路径
    ir_dir = os.path.join(ORIGINAL_DATA_ROOT, "Ir")
    vis_dir = os.path.join(ORIGINAL_DATA_ROOT, "Vis")

    # 获取所有文件名（仅保留前缀，忽略后缀）
    ir_filenames = [f for f in os.listdir(ir_dir) if os.path.splitext(f)[-1].lower() in IMG_SUFFIX]
    vis_filenames = [f for f in os.listdir(vis_dir) if os.path.splitext(f)[-1].lower() in IMG_SUFFIX]

    ir_basenames = set([os.path.splitext(f)[0] for f in ir_filenames])
    vis_basenames = set([os.path.splitext(f)[0] for f in vis_filenames])

    # 筛选成对的文件名前缀
    common_basenames = list(ir_basenames & vis_basenames)
    common_basenames.sort()

    # 为每个前缀匹配具体的文件路径和后缀
    paired_files = []
    for basename in common_basenames:
        # 匹配IR文件路径
        ir_path = None
        ir_suffix = None
        for suffix in IMG_SUFFIX:
            candidate = os.path.join(ir_dir, basename + suffix)
            if os.path.exists(candidate):
                ir_path = candidate
                ir_suffix = suffix
                break

        # 匹配VIS文件路径
        vis_path = None
        vis_suffix = None
        for suffix in IMG_SUFFIX:
            candidate = os.path.join(vis_dir, basename + suffix)
            if os.path.exists(candidate):
                vis_path = candidate
                vis_suffix = suffix
                break

        if ir_path and vis_path:
            paired_files.append({
                "basename": basename,
                "ir_path": ir_path,
                "ir_suffix": ir_suffix,
                "vis_path": vis_path,
                "vis_suffix": vis_suffix
            })

    print(f"✅ 找到 {len(paired_files)} 对有效IR/VIS图像")
    if len(paired_files) == 0:
        raise ValueError("❌ 未找到成对的红外/可见光图像！请检查文件名是否一致")
    return paired_files


def crop_image_to_patches(img, patch_size=PATCH_SIZE, padding_mode=PADDING_MODE):
    """
    将单张图像裁剪为256x256的Patch
    :param img: 输入图像（np.ndarray）
    :param patch_size: Patch尺寸 (H, W)
    :param padding_mode: 边界处理方式 "zero"/"trunc"
    :return: (patches, patch_coords) -> Patch列表 + 每个Patch的坐标信息（用于命名）
    """
    img_h, img_w = img.shape[:2]
    patch_h, patch_w = patch_size

    # 计算分块的行数和列数
    num_rows = np.ceil(img_h / patch_h).astype(int)
    num_cols = np.ceil(img_w / patch_w).astype(int)

    patches = []
    patch_coords = []  # 保存每个Patch的行/列索引（用于命名）

    for row in range(num_rows):
        for col in range(num_cols):
            # 计算当前Patch的坐标
            y_start = row * patch_h
            y_end = y_start + patch_h
            x_start = col * patch_w
            x_end = x_start + patch_w

            # 提取Patch
            patch = img[y_start:y_end, x_start:x_end]

            # 边界处理：补零（保证Patch尺寸严格256x256）
            if padding_mode == "zero":
                pad_h = patch_h - patch.shape[0] if patch.shape[0] < patch_h else 0
                pad_w = patch_w - patch.shape[1] if patch.shape[1] < patch_w else 0
                if pad_h > 0 or pad_w > 0:
                    pad_width = [(0, pad_h), (0, pad_w)]
                    if len(patch.shape) == 3:
                        pad_width.append((0, 0))  # 3通道图像补零
                    patch = np.pad(patch, pad_width, mode="constant", constant_values=0)

            patches.append(patch)
            patch_coords.append((row, col))

    return patches, patch_coords


def save_patches(patches, patch_coords, save_dir, basename, suffix, is_ir=False):
    """
    保存裁剪后的Patch，命名规则：basename_patch_行索引_列索引.suffix
    :param patches: Patch列表
    :param patch_coords: 每个Patch的(行, 列)索引
    :param save_dir: Patch保存目录
    :param basename: 原文件前缀
    :param suffix: 原文件后缀
    :param is_ir: 是否为红外图像（特殊处理单通道）
    """
    for idx, (patch, (row, col)) in enumerate(zip(patches, patch_coords)):
        # 生成Patch文件名：原前缀_patch_行_列.原后缀（如：img001_patch_00_00.png）
        patch_filename = f"{basename}_patch_{row:02d}_{col:02d}{suffix}"
        patch_path = os.path.join(save_dir, patch_filename)

        # 保存红外图像（单通道灰度图）
        if is_ir and SAVE_IR_AS_GRAY:
            if len(patch.shape) == 3:
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(patch_path, patch)
        # 保存可见光图像（3通道RGB）
        else:
            # 兼容PIL和OpenCV的通道顺序
            if len(patch.shape) == 3 and patch.shape[2] == 3:
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            cv2.imwrite(patch_path, patch)


# ===================== 主处理函数 =====================
def process_dataset():
    """主流程：裁剪IR/VIS成对图像为Patch，保持结构和对应关系"""
    # 1. 创建目录结构
    ir_patch_dir, vis_patch_dir = create_dir_structure()
    print(f"📁 新建Patch数据集目录：{PATCH_DATA_ROOT}")
    print(f"   - IR Patch保存路径：{ir_patch_dir}")
    print(f"   - VIS Patch保存路径：{vis_patch_dir}")

    # 2. 获取成对文件列表
    paired_files = get_paired_filenames()

    # 3. 批量处理每对图像
    total_patches = 0
    pbar = tqdm(paired_files, desc="处理图像对生成Patch")
    for file_info in pbar:
        basename = file_info["basename"]
        ir_path = file_info["ir_path"]
        vis_path = file_info["vis_path"]
        ir_suffix = file_info["ir_suffix"]
        vis_suffix = file_info["vis_suffix"]

        # 读取图像（保持原始通道信息）
        # IR图像：优先用PIL读取（兼容不同格式），保留原始通道
        ir_img = Image.open(ir_path)
        if ir_img.mode == "L":
            ir_img = np.array(ir_img)  # 单通道灰度图
        else:
            ir_img = np.array(ir_img.convert("RGB"))  # 转为3通道（兼容训练代码）

        # VIS图像：转为RGB
        vis_img = np.array(Image.open(vis_path).convert("RGB"))

        # 裁剪为Patch（IR和VIS使用完全相同的分块规则，保证一一对应）
        ir_patches, ir_coords = crop_image_to_patches(ir_img)
        vis_patches, vis_coords = crop_image_to_patches(vis_img)

        # 校验：IR和VIS的Patch数量必须一致（保证成对）
        assert len(ir_patches) == len(vis_patches), \
            f"❌ {basename}的IR/VIS Patch数量不一致！IR:{len(ir_patches)}, VIS:{len(vis_patches)}"

        # 保存Patch
        save_patches(ir_patches, ir_coords, ir_patch_dir, basename, ir_suffix, is_ir=True)
        save_patches(vis_patches, vis_coords, vis_patch_dir, basename, vis_suffix, is_ir=False)

        # 统计总Patch数
        total_patches += len(ir_patches)
        pbar.set_postfix({"单图Patch数": len(ir_patches), "累计Patch数": total_patches})

    # 4. 输出处理结果
    print("\n🎉 数据集分块完成！")
    print(f"📊 统计信息：")
    print(f"   - 处理图像对数量：{len(paired_files)}")
    print(f"   - 生成总Patch对数：{total_patches}")
    print(f"   - Patch尺寸：{PATCH_SIZE[0]}×{PATCH_SIZE[1]}")
    print(f"   - 边界处理方式：{PADDING_MODE}")
    print(f"   - 新数据集路径：{PATCH_DATA_ROOT}")


# ===================== 执行入口 =====================
if __name__ == "__main__":
    try:
        process_dataset()
    except Exception as e:
        print(f"\n❌ 处理失败：{str(e)}")
        raise