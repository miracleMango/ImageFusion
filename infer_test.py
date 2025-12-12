import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.model import ImageFusionNetworkWithSourcePE  # 导入你的模型


# -------------------------- 配置参数（简化版本） --------------------------
WEIGHT_PATH = "./saved_models/best_model_loss_0.1657.pth"
TESTSET_ROOT = "./datasets/M3FD_Fusion_test"  # 【修改为你的测试集根目录】
IR_SUB_DIR = "Ir"  # 红外子目录
VIS_SUB_DIR = "Vis"  # 可见光子目录
OUTPUT_DIR = "./results/fusion_results_test"
PATCH_SIZE = (128, 128)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------- 核心工具函数（简化版本） --------------------------
def split_image_to_patches(img, patch_size=PATCH_SIZE, is_ir=False):
    """简单分块，不重叠"""
    img_np = np.array(img)
    img_h, img_w = img_np.shape[:2]
    patch_h, patch_w = patch_size
    
    # 计算行数和列数
    num_rows = img_h // patch_h + (1 if img_h % patch_h > 0 else 0)
    num_cols = img_w // patch_w + (1 if img_w % patch_w > 0 else 0)
    
    patches = []
    coords = []
    
    for row in range(num_rows):
        for col in range(num_cols):
            y_start = row * patch_h
            x_start = col * patch_w
            y_end = min(y_start + patch_h, img_h)
            x_end = min(x_start + patch_w, img_w)
            
            patch = img_np[y_start:y_end, x_start:x_end]
            
            # 填充到固定大小
            pad_h = patch_h - patch.shape[0] if patch.shape[0] < patch_h else 0
            pad_w = patch_w - patch.shape[1] if patch.shape[1] < patch_w else 0
            if pad_h > 0 or pad_w > 0:
                if is_ir:
                    pad_width = [(0, pad_h), (0, pad_w)]
                else:
                    pad_width = [(0, pad_h), (0, pad_w), (0, 0)]
                patch = np.pad(patch, pad_width, mode="constant", constant_values=0)
            
            # 转换为PIL图像
            if is_ir:
                patch_img = Image.fromarray(patch, mode="L")
            else:
                patch_img = Image.fromarray(patch, mode="RGB")
            patches.append(patch_img)
            coords.append((y_start, y_end, x_start, x_end))
    
    return patches, coords, (img_h, img_w)


def merge_patches_to_full(patches, coords, original_size, is_ir=False):
    """简单拼接，直接放置"""
    original_h, original_w = original_size
    
    # 初始化结果数组
    if is_ir:
        full_img = np.zeros((original_h, original_w), dtype=np.uint8)
    else:
        full_img = np.zeros((original_h, original_w, 3), dtype=np.uint8)
    
    for patch, (y_s, y_e, x_s, x_e) in zip(patches, coords):
        patch_np = np.array(patch, dtype=np.uint8)
        h_patch = y_e - y_s
        w_patch = x_e - x_s
        
        # 裁剪回原始大小（去除填充部分）
        patch_cropped = patch_np[:h_patch, :w_patch]
        
        # 直接放置到对应位置
        if is_ir:
            full_img[y_s:y_e, x_s:x_e] = patch_cropped
        else:
            full_img[y_s:y_e, x_s:x_e, :] = patch_cropped
    
    # 转换为PIL图像
    if is_ir:
        return Image.fromarray(full_img, mode="L")
    else:
        return Image.fromarray(full_img)


# 拆分transform：保持不变
def get_ir_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def get_vis_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def post_process(tensor, is_ir=False):
    """后处理函数"""
    tensor = tensor.detach().cpu()
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0.0, 1.0)

    if is_ir:
        img = tensor.squeeze(0).numpy()
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img, mode="L")
    else:
        img = tensor.permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img)


# -------------------------- 单张图片对推理函数（简化版本） --------------------------
def infer_single_pair(ir_path, vis_path, output_dir):
    basename = os.path.basename(ir_path)
    fusion_path = os.path.join(output_dir, basename)  # 直接在输出目录，使用原始文件名

    # 加载模型（仅加载一次）
    global model
    if "model" not in globals():
        model = ImageFusionNetworkWithSourcePE(
            vis_img_channels=3,
            ir_img_channels=1,
            feature_channels=64,
            num_heads=16,
            use_position_encoding=True
        ).to(DEVICE)
        checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"✅ 模型已加载，开始处理图片对：{basename}")

    # 读取图像
    ir_img_full = Image.open(ir_path).convert("L")
    vis_img_full = Image.open(vis_path).convert("RGB")

    # 简单分块
    ir_patches, coords, original_size = split_image_to_patches(ir_img_full, is_ir=True)
    vis_patches, _, _ = split_image_to_patches(vis_img_full, is_ir=False)

    # 加载transform
    ir_transform = get_ir_transform()
    vis_transform = get_vis_transform()

    fusion_patches = []
    with torch.no_grad():
        for ir_patch, vis_patch in zip(ir_patches, vis_patches):
            # 预处理
            ir_tensor = ir_transform(ir_patch).unsqueeze(0).to(DEVICE)
            vis_tensor = vis_transform(vis_patch).unsqueeze(0).to(DEVICE)

            # 强制修正红外通道数和尺寸
            if ir_tensor.shape[1] != 1:
                ir_tensor = ir_tensor[:, 0:1, :, :]
            if ir_tensor.shape[2:] != (128, 128):
                ir_tensor = torch.nn.functional.interpolate(ir_tensor, size=(128, 128), mode="nearest")

            # 模型推理
            outputs = model(ir_tensor, vis_tensor)

            # 后处理
            fusion_patches.append(post_process(outputs["img_fusion_pred"][0], is_ir=False))

    # 简单拼接
    full_fusion = merge_patches_to_full(fusion_patches, coords, original_size, is_ir=False)

    # 保存融合图
    full_fusion.save(fusion_path)
    
    return basename, fusion_path


# -------------------------- 批量处理测试集 --------------------------
def batch_process_testset():
    ir_dir = os.path.join(TESTSET_ROOT, IR_SUB_DIR)
    vis_dir = os.path.join(TESTSET_ROOT, VIS_SUB_DIR)
    ir_filenames = [f for f in os.listdir(ir_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    vis_filenames = [f for f in os.listdir(vis_dir) if f.endswith((".png", ".jpg", ".jpeg"))]

    common_filenames = list(set(ir_filenames) & set(vis_filenames))
    if not common_filenames:
        print("❌ 未找到匹配的红外-可见光图片对！请检查文件名。")
        return
    print(f"✅ 找到 {len(common_filenames)} 对匹配的图片，开始批量处理...")

    for idx, filename in enumerate(common_filenames, 1):
        ir_path = os.path.join(ir_dir, filename)
        vis_path = os.path.join(vis_dir, filename)
        basename, fusion_path = infer_single_pair(ir_path, vis_path, OUTPUT_DIR)
        print(f"🔧 已完成 {idx}/{len(common_filenames)}：{basename}，融合图保存至：{fusion_path}")

    print(f"\n✅ 批量处理完成！")
    print(f"   - 所有融合图保存至：{OUTPUT_DIR}")


# -------------------------- 运行 --------------------------
if __name__ == "__main__":
    batch_process_testset()