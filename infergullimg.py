import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.model import ImageFusionNetworkWithSourcePE  # 导入你的模型


# -------------------------- 配置参数（简化版本，删除PATCH_SIZE） --------------------------
WEIGHT_PATH = "./saved_models/best_model_loss_0.0712.pth"
TESTSET_ROOT = "./datasets/M3FD_Fusion_test"  # 【修改为你的测试集根目录】
IR_SUB_DIR = "Ir"  # 红外子目录
VIS_SUB_DIR = "Vis"  # 可见光子目录
OUTPUT_DIR = "./results/fusion_results_test_color_1.0_fullimg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------- 核心工具函数（删除分块/合并函数，保留关键transform和后处理） --------------------------
# 保持transform逻辑不变（仅删除Resize，因为整图推理不需要固定尺寸）
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
    """后处理函数（适配整图输出，逻辑不变）"""
    tensor = tensor.detach().cpu()
    tensor = (tensor + 1.0) / 2.0  # 反归一化到[0,1]
    tensor = torch.clamp(tensor, 0.0, 1.0)  # 防止数值溢出

    if is_ir:
        img = tensor.squeeze(0).numpy()
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img, mode="L")
    else:
        img = tensor.permute(1, 2, 0).numpy()  # (C,H,W) -> (H,W,C)
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img)


# -------------------------- 单张图片对推理函数（整图推理版本） --------------------------
def infer_single_pair(ir_path, vis_path, output_dir):
    basename = os.path.basename(ir_path)
    fusion_path = os.path.join(output_dir, basename)  # 输出文件名与输入保持一致

    # 加载模型（全局仅加载一次，避免重复加载）
    global model
    if "model" not in globals():
        model = ImageFusionNetworkWithSourcePE(
            vis_img_channels=3,
            ir_img_channels=1,
            feature_channels=64,
            num_heads=16,
            use_position_encoding=True
        ).to(DEVICE)
        
        # 加载模型权重（兼容完整checkpoint或仅state_dict）
        checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()  # 切换到评估模式
        print(f"✅ 模型已加载，开始处理图片对：{basename}")

    # 1. 读取整张图片（不做分块）
    ir_img_full = Image.open(ir_path).convert("L")  # 红外图转单通道
    vis_img_full = Image.open(vis_path).convert("RGB")  # 可见光图转三通道

    # 2. 加载transform并预处理整张图片
    ir_transform = get_ir_transform()
    vis_transform = get_vis_transform()
    
    ir_tensor = ir_transform(ir_img_full).unsqueeze(0).to(DEVICE)  # (1,1,H,W)
    vis_tensor = vis_transform(vis_img_full).unsqueeze(0).to(DEVICE)  # (1,3,H,W)

    # 3. 整图推理（无patch循环）
    with torch.no_grad():  # 禁用梯度计算，节省显存
        outputs = model(ir_tensor, vis_tensor)
        fusion_tensor = outputs["img_fusion_pred"][0]  # 取batch中第一个（仅单张）

    # 4. 后处理整张融合图
    full_fusion = post_process(fusion_tensor, is_ir=False)

    # 5. 保存整图融合结果
    full_fusion.save(fusion_path)
    
    return basename, fusion_path


# -------------------------- 批量处理测试集（逻辑不变，仅调用整图推理函数） --------------------------
def batch_process_testset():
    ir_dir = os.path.join(TESTSET_ROOT, IR_SUB_DIR)
    vis_dir = os.path.join(TESTSET_ROOT, VIS_SUB_DIR)
    
    # 获取匹配的文件名（红外和可见光文件名需一致）
    ir_filenames = [f for f in os.listdir(ir_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    vis_filenames = [f for f in os.listdir(vis_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    common_filenames = list(set(ir_filenames) & set(vis_filenames))

    if not common_filenames:
        print("❌ 未找到匹配的红外-可见光图片对！请检查文件名是否一致。")
        return
    
    print(f"✅ 找到 {len(common_filenames)} 对匹配的图片，开始批量整图推理...")

    for idx, filename in enumerate(common_filenames, 1):
        ir_path = os.path.join(ir_dir, filename)
        vis_path = os.path.join(vis_dir, filename)
        basename, fusion_path = infer_single_pair(ir_path, vis_path, OUTPUT_DIR)
        print(f"🔧 已完成 {idx}/{len(common_filenames)}：{basename}，融合图保存至：{fusion_path}")

    print(f"\n✅ 批量整图推理完成！")
    print(f"   - 所有融合图保存至：{os.path.abspath(OUTPUT_DIR)}")


# -------------------------- 运行 --------------------------
if __name__ == "__main__":
    batch_process_testset()
