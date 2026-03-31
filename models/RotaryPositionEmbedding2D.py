import torch
import torch.nn as nn
import math

def find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=64):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=64):
    low = math.floor(find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1)

def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

def get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0

class RotaryPositionEmbedding2D(nn.Module):
    def __init__(
        self, 
        head_dim: int, 
        theta: float = 10000.0, 
        original_train_size: tuple = (64, 64),  # 请确保这里是你实际训练时的特征图尺寸 (H, W)
        extrapolation_factor: float = 1.0,
        attn_factor: float = 1.0,
        beta_fast: int = 32,
        beta_slow: int = 1,
        finetuned: bool = False
    ):
        super().__init__()
        assert head_dim % 4 == 0, f"二维RoPE的head_dim必须是4的倍数，当前输入为{head_dim}"
        
        self.head_dim = head_dim
        self.theta = theta
        self.freq_len = self.head_dim // 4
        
        # YaRN 核心参数
        self.original_h, self.original_w = original_train_size
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        # 初始化逆频率
        self._init_inv_freqs(finetuned)

        # ======================== 关键修改1：缓存变量重命名，去掉"max" ========================
        self.cached_h = self.original_h
        self.cached_w = self.original_w
        self._update_cos_sin_cache(self.original_h, self.original_w, device=None, dtype=torch.get_default_dtype())

    def _init_inv_freqs(self, finetuned):
        pos_freqs = self.theta ** (torch.arange(0, self.head_dim, 4).float() / self.head_dim)
        
        if finetuned:
            self.inv_freq_x = 1.0 / pos_freqs
            self.inv_freq_y = 1.0 / pos_freqs
            self.mscale_x = 1.0
            self.mscale_y = 1.0
        else:
            self.register_buffer("inv_freq_x", 1.0 / pos_freqs, persistent=False)
            self.register_buffer("inv_freq_y", 1.0 / pos_freqs, persistent=False)
            self.mscale_x = 1.0
            self.mscale_y = 1.0

    def _yarn_1d(self, scale, original_max_len, device):
        dim_for_correction = self.head_dim // 2
        pos_freqs = self.theta ** (torch.arange(0, self.head_dim, 4).float().to(device) / self.head_dim)
        
        inv_freq_extrap = 1.0 / pos_freqs
        inv_freq_interp = 1.0 / (scale * pos_freqs)
        
        low, high = find_correction_range(
            self.beta_fast, self.beta_slow, 
            dim=dim_for_correction, 
            base=self.theta, 
            max_position_embeddings=original_max_len
        )
        
        inv_freq_mask = (1 - linear_ramp_mask(low, high, self.freq_len).float().to(device)) * self.extrapolation_factor
        inv_freq = inv_freq_interp * (1 - inv_freq_mask) + inv_freq_extrap * inv_freq_mask
        mscale = float(get_mscale(scale) * self.attn_factor)
        
        return inv_freq, mscale

    def _update_cos_sin_cache(self, H: int, W: int, device, dtype):
        if device is None:
            device = self.inv_freq_x.device

        y_coords = torch.arange(H, device=device, dtype=torch.float32)
        x_coords = torch.arange(W, device=device, dtype=torch.float32)
        
        freqs_x = torch.outer(x_coords, self.inv_freq_x.float())
        freqs_y = torch.outer(y_coords, self.inv_freq_y.float())
        
        freqs_x_expanded = freqs_x[None, :, :].expand(H, W, self.freq_len)
        freqs_y_expanded = freqs_y[:, None, :].expand(H, W, self.freq_len)
        
        freqs_x_flat = freqs_x_expanded.reshape(H * W, self.freq_len)
        freqs_y_flat = freqs_y_expanded.reshape(H * W, self.freq_len)
        
        cos_x = (freqs_x_flat.cos() * self.mscale_x).to(dtype)
        sin_x = (freqs_x_flat.sin() * self.mscale_x).to(dtype)
        cos_y = (freqs_y_flat.cos() * self.mscale_y).to(dtype)
        sin_y = (freqs_y_flat.sin() * self.mscale_y).to(dtype)
        
        self.register_buffer("cos_x_cached", cos_x[None, None, :, :], persistent=False)
        self.register_buffer("sin_x_cached", sin_x[None, None, :, :], persistent=False)
        self.register_buffer("cos_y_cached", cos_y[None, None, :, :], persistent=False)
        self.register_buffer("sin_y_cached", sin_y[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        batch_size, n_heads, seq_len, head_dim = x.shape
        assert head_dim == self.head_dim
        assert H * W == seq_len

        # ======================== 关键修改2：只要尺寸不一致就更新缓存（无论变大变小） ========================
        if H != self.cached_h or W != self.cached_w:
            self.cached_h = H
            self.cached_w = W
            
            scale_h = H / self.original_h
            scale_w = W / self.original_w
            
            self.inv_freq_x, self.mscale_x = self._yarn_1d(scale_w, self.original_w, x.device)
            self.inv_freq_y, self.mscale_y = self._yarn_1d(scale_h, self.original_h, x.device)
            
            self._update_cos_sin_cache(H, W, x.device, x.dtype)

        cos_x = self.cos_x_cached.to(x.dtype)
        sin_x = self.sin_x_cached.to(x.dtype)
        cos_y = self.cos_y_cached.to(x.dtype)
        sin_y = self.sin_y_cached.to(x.dtype)

        x1, x2, y1, y2 = x[..., 0::4], x[..., 1::4], x[..., 2::4], x[..., 3::4]
        
        x1_rot = x1 * cos_x - x2 * sin_x
        x2_rot = x1 * sin_x + x2 * cos_x
        y1_rot = y1 * cos_y - y2 * sin_y
        y2_rot = y1 * sin_y + y2 * cos_y

        x_rotated = torch.cat([x1_rot, x2_rot, y1_rot, y2_rot], dim=-1)
        
        return x_rotated
