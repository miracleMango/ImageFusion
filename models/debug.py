def print_channel_distribution(tensor, name, step, is_rgb=False):
    """
    打印张量的通道分布（修复变量未定义错误）
    Args:
        tensor: 输入张量 [B, C, H, W]
        name: 张量名称（用于日志）
        step: 当前调试step
        is_rgb: 是否为RGB三通道张量（自动提取前3通道）
    """
    if tensor is None:
        return

    device = tensor.device
    tensor = tensor.detach().cpu()  # 避免影响梯度
    b, c, h, w = tensor.shape

    print(f"\n===== Step {step} | {name} =====")
    print(f"形状: {tensor.shape}, 设备: {device}")

    # 1. 整体统计
    print(
        f"整体 - 最小值: {tensor.min():.4f}, 最大值: {tensor.max():.4f}, 均值: {tensor.mean():.4f}, 方差: {tensor.var():.4f}")

    # 2. RGB通道专项统计（重点修复：先收集所有通道均值，再计算比值）
    if is_rgb and c >= 3:
        rgb_means = []  # 存储R/G/B均值
        rgb_mins = []  # 存储R/G/B最小值
        rgb_maxs = []  # 存储R/G/B最大值
        rgb_vars = []  # 存储R/G/B方差

        # 第一步：先收集所有RGB通道的统计信息
        for idx, ch_name in enumerate(['R', 'G', 'B']):
            ch = tensor[:, idx, :, :]
            ch_mean = ch.mean()
            ch_min = ch.min()
            ch_max = ch.max()
            ch_var = ch.var()

            rgb_means.append(ch_mean)
            rgb_mins.append(ch_min)
            rgb_maxs.append(ch_max)
            rgb_vars.append(ch_var)

            print(
                f"  {ch_name}通道 - 最小值: {ch_min:.4f}, 最大值: {ch_max:.4f}, 均值: {ch_mean:.4f}, 方差: {ch_var:.4f}")

        # 第二步：统一计算比值（此时g_mean已定义）
        r_mean, g_mean, b_mean = rgb_means
        r_g_ratio = r_mean / (g_mean + 1e-8)
        b_g_ratio = b_mean / (g_mean + 1e-8)
        print(f"  R/G比值: {r_g_ratio:.4f} (正常应接近1)")
        print(f"  B/G比值: {b_g_ratio:.4f} (正常应接近1)")

        # 额外判断是否偏绿
        if g_mean > r_mean * 1.5 or g_mean > b_mean * 1.5:
            print(f"  ❌ 警告：G通道均值({g_mean:.4f})远大于R/B通道，存在偏绿风险！")

    # 3. 单通道（红外）统计
    elif c == 1:
        ch = tensor[:, 0, :, :]
        print(f"  单通道 - 最小值: {ch.min():.4f}, 最大值: {ch.max():.4f}, 均值: {ch.mean():.4f}, 方差: {ch.var():.4f}")

    # 4. 特征通道前8个统计（看分组卷积是否均衡）
    else:
        print("  前8个特征通道均值:")
        for i in range(min(8, c)):
            print(f"    通道{i}: {tensor[:, i, :, :].mean():.4f}")