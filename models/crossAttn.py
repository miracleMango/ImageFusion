from models.SourcePositionalEncoding import SourcePositionalEncoding2D
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttentionWithSourcePE(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, use_position_encoding=True,
                 src_q_channels=1, src_k_channels=1, src_v_channels=1):  # 新增源图像通道参数
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = (hidden_size * 2) // num_heads
        self.use_position_encoding = use_position_encoding

        assert self.head_dim * num_heads == hidden_size * 2, "hidden_size必须是num_heads的整数倍"

        # 关键修改：不再动态初始化，而是__init__里显式创建（注册为子模块）
        self.pos_encoding_q = SourcePositionalEncoding2D(src_q_channels, self.hidden_size * 2) if use_position_encoding else None
        self.pos_encoding_k = SourcePositionalEncoding2D(src_k_channels, self.hidden_size * 2) if use_position_encoding else None
        self.pos_encoding_v = SourcePositionalEncoding2D(src_v_channels, self.hidden_size * 2) if use_position_encoding else None

        # 交叉注意力投影层（原逻辑保留）
        self.w_q = nn.Linear(hidden_size * 2, hidden_size * 2, bias=True)
        self.w_k = nn.Linear(hidden_size * 2, hidden_size * 2, bias=True)
        self.w_v = nn.Linear(hidden_size * 2, hidden_size * 2, bias=True)
        self.proj = nn.Linear(hidden_size * 2, hidden_size * 2, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('scale', torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))

    def forward(self, fq, fk, fv, source_img_q, source_img_k, source_img_v):
        batch_size, c, h, w = fq.shape
        seq_len = h * w

        # 步骤1：添加位置编码（无需动态初始化，已在__init__里创建）
        if self.use_position_encoding:
            fq = self.pos_encoding_q(fq, source_img_q)
            fk = self.pos_encoding_k(fk, source_img_k)
            fv = self.pos_encoding_v(fv, source_img_v)

        # 后续逻辑完全保留
        fq = fq.reshape(batch_size, c, seq_len).permute(0, 2, 1)
        fk = fk.reshape(batch_size, c, seq_len).permute(0, 2, 1)
        fv = fv.reshape(batch_size, c, seq_len).permute(0, 2, 1)

        q = self.w_q(fq)
        k = self.w_k(fk)
        v = self.w_v(fv)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
            scale=self.scale
        )

        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size * 2)
        output = self.proj(attn_output)
        output = output.permute(0, 2, 1).reshape(batch_size, c, h, w)
        output = self.dropout(output)

        return output, None


class MultiHeadCrossAttentionFusionWithSourcePE(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, use_position_encoding=True,
                 src_qk_channels=4, src_v_channels=3):  # 新增源图像通道参数（fusion分支qk是4通道，v是3通道）
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = (hidden_size * 2) // num_heads
        self.use_position_encoding = use_position_encoding

        assert self.head_dim * num_heads == hidden_size * 2, "hidden_size必须是num_heads的整数倍"

        # 关键修改：__init__里显式初始化pos_encoding
        self.pos_encoding_q = SourcePositionalEncoding2D(src_qk_channels, self.hidden_size * 4) if use_position_encoding else None
        self.pos_encoding_k = SourcePositionalEncoding2D(src_qk_channels, self.hidden_size * 4) if use_position_encoding else None
        self.pos_encoding_v = SourcePositionalEncoding2D(src_v_channels, self.hidden_size * 2) if use_position_encoding else None

        # 投影层（原逻辑保留）
        self.w_q = nn.Linear(hidden_size * 4, hidden_size * 2, bias=True)
        self.w_k = nn.Linear(hidden_size * 4, hidden_size * 2, bias=True)
        self.w_v = nn.Linear(hidden_size * 2, hidden_size * 2, bias=True)
        self.proj = nn.Linear(hidden_size * 2, hidden_size * 2, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('scale', torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))

    def forward(self, fq, fk, fv, source_img_q, source_img_k, source_img_v):
        batch_size, c_v, h, w = fv.shape
        _, c_qk, _, _ = fk.shape
        seq_len = h * w

        # 步骤1：添加位置编码（无需动态初始化）
        if self.use_position_encoding:
            fq = self.pos_encoding_q(fq, source_img_q)
            fk = self.pos_encoding_k(fk, source_img_k)
            fv = self.pos_encoding_v(fv, source_img_v)

        # 后续逻辑完全保留
        fq = fq.reshape(batch_size, c_qk, seq_len).permute(0, 2, 1)
        fk = fk.reshape(batch_size, c_qk, seq_len).permute(0, 2, 1)
        fv = fv.reshape(batch_size, c_v, seq_len).permute(0, 2, 1)

        q = self.w_q(fq)
        k = self.w_k(fk)
        v = self.w_v(fv)

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
            scale=self.scale
        )

        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size * 2)
        output = self.proj(attn_output)
        output = output.permute(0, 2, 1).reshape(batch_size, self.hidden_size * 2, h, w)
        output = self.dropout(output)

        return output, None