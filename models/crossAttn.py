from models.RotaryPositionEmbedding2D import RotaryPositionEmbedding2D
import torch
import torch.nn as nn
import torch.nn.functional as F

# 保持原有ResidualCrossAttentionBlock完全不变（仅补充少量通道校验）
class ResidualCrossAttentionBlock(nn.Module):
    """通用的残差交叉注意力块（未做核心修改，仅补充通道校验）"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1,
                q_channels=None, k_channels=None, v_channels=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = (hidden_size * 2) // num_heads
        self.q_channels = q_channels if q_channels is not None else hidden_size * 2
        self.k_channels = k_channels if k_channels is not None else hidden_size * 2
        self.v_channels = v_channels if v_channels is not None else hidden_size * 2

        assert self.head_dim * self.num_heads == hidden_size * 2, \
            f"head_dim({self.head_dim}) * num_heads({self.num_heads}) must equal hidden_size*2({hidden_size*2})"
        assert self.head_dim % 4 == 0, \
            f"head_dim({self.head_dim}) must be a multiple of 4 for 2D RoPE"

        # 投影层（保持不变，适配hidden_size*2的通道数）
        self.w_q = nn.Linear(self.q_channels, hidden_size * 2, bias=True)
        self.w_k = nn.Linear(self.k_channels, hidden_size * 2, bias=True)
        self.w_v = nn.Linear(self.v_channels, hidden_size * 2, bias=True)
        self.proj = nn.Linear(hidden_size * 2, hidden_size * 2, bias=True)
        
        # 位置编码（在外部传入，这里只定义但不创建）
        self.pos_encoding_q = None
        self.pos_encoding_k = None
        self.pos_encoding_v = None
        
        # Dropout和归一化（保持不变）
        self.dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        
        # 层归一化（保持不变）
        self.norm1 = nn.LayerNorm(hidden_size * 2)  # 注意力后的归一化
        self.norm2 = nn.LayerNorm(hidden_size * 2)  # 前馈后的归一化
        
        # 前馈网络（保持不变）
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.Dropout(dropout)
        )
        
        # 可学习的残差权重（保持不变）
        self.residual_weight = nn.Parameter(torch.tensor(1.0))
        
        self.register_buffer('scale', torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
    
    def set_position_encodings(self, pos_encoding_q, pos_encoding_k, pos_encoding_v):
        """设置位置编码（由外部传入，保持不变）"""
        self.pos_encoding_q = pos_encoding_q
        self.pos_encoding_k = pos_encoding_k
        self.pos_encoding_v = pos_encoding_v
    
    def forward(self, fq, fk, fv, use_position_encoding=True):
        batch_size, c_q, h, w = fq.shape
        seq_len = h * w
        
        # 保存原始fq用于残差连接（保持不变）
        residual = fq.clone()
        
        # 将特征重塑为序列（保持不变，确保维度对齐）
        fq_seq = fq.reshape(batch_size, c_q, seq_len).permute(0, 2, 1)
        fk_seq = fk.reshape(batch_size, self.k_channels, seq_len).permute(0, 2, 1)
        fv_seq = fv.reshape(batch_size, self.v_channels, seq_len).permute(0, 2, 1)
        
        # 投影（保持不变）
        q = self.w_q(fq_seq)
        k = self.w_k(fk_seq)
        v = self.w_v(fv_seq)
        
        # 多头处理（保持不变，确保张量连续）
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        if use_position_encoding and self.pos_encoding_q is not None:
            q = self.pos_encoding_q(q, h, w)
            k = self.pos_encoding_k(k, h, w)
        
        # 注意力计算（保持不变，使用PyTorch原生Scaled Dot Product Attention）
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
            scale=self.scale
        )
        
        # 重组输出（保持不变）
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size * 2)
        output_seq = self.proj(attn_output)
        
        # 第一个残差连接（保持不变）
        output_seq = output_seq + self.residual_weight * self.residual_dropout(
            residual.reshape(batch_size, c_q, seq_len).permute(0, 2, 1)
        )
        
        # 层归一化（保持不变）
        output_seq = self.norm1(output_seq)
        
        # 前馈网络（保持不变）
        ffn_residual = output_seq.clone()
        ffn_output = self.ffn(output_seq)
        
        # 第二个残差连接（保持不变）
        output_seq = ffn_residual + ffn_output
        
        # 最终层归一化（保持不变）
        output_seq = self.norm2(output_seq)
        
        # 转换回4D特征图（保持不变，确保维度与输入一致）
        output = output_seq.permute(0, 2, 1).reshape(batch_size, c_q, h, w)
        
        return output

# 交叉注意力外层模块（修改：适配SourcePositionalEncoding2D，默认src通道为3）
class MultiHeadCrossAttentionWithSourcePE(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, use_position_encoding=True, num_blocks=1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_position_encoding = use_position_encoding
        self.num_blocks = num_blocks

        if use_position_encoding:
            self.pos_encoding_q = RotaryPositionEmbedding2D(
                head_dim = (hidden_size*2)//num_heads
            )
            self.pos_encoding_k = RotaryPositionEmbedding2D(
                head_dim = (hidden_size*2)//num_heads
            )
            self.pos_encoding_v = RotaryPositionEmbedding2D(
                head_dim = (hidden_size*2)//num_heads
            )
        else:
            self.pos_encoding_q = None
            self.pos_encoding_k = None
            self.pos_encoding_v = None
        
        # 创建多个残差注意力块（保持不变）
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = ResidualCrossAttentionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                q_channels=hidden_size * 2,
                k_channels=hidden_size * 2,
                v_channels=hidden_size * 2
            )
            
            if use_position_encoding:
                block.set_position_encodings(
                    self.pos_encoding_q,
                    self.pos_encoding_k,
                    self.pos_encoding_v
                )
            
            self.blocks.append(block)
        
        # 最终的归一化和dropout（保持不变）
        self.final_norm = nn.LayerNorm(hidden_size * 2)
        self.final_dropout = nn.Dropout(dropout)
        
    def forward(self, fq, fk, fv):
        # 保存原始输入用于最终残差连接（保持不变）
        original_input = fq.clone()
        
        # 存储中间特征（可选，保持不变）
        intermediate_outputs = []
        
        # 逐个处理残差块（保持不变）
        for i, block in enumerate(self.blocks):
            fq = block(
                fq, fk, fv, 
                use_position_encoding=self.use_position_encoding
            )
            
            # 保存中间输出（如果需要，保持不变）
            if self.training:
                intermediate_outputs.append(fq.clone())
        
        # 最终归一化（保持不变，确保维度对齐）
        batch_size, c, h, w = fq.shape
        seq_len = h * w
        
        fq_seq = fq.reshape(batch_size, c, seq_len).permute(0, 2, 1)
        fq_seq = self.final_norm(fq_seq)
        fq = fq_seq.permute(0, 2, 1).reshape(batch_size, c, h, w)
        
        # 最终的全局残差连接（保持不变）
        fq = fq + original_input
        
        # 最终的dropout（保持不变）
        fq = self.final_dropout(fq)
        
        # 保持返回格式：训练时返回中间输出，测试时返回None（适配主模型调用）
        return fq, intermediate_outputs if self.training else None

# 自注意力外层模块（修改：适配SourcePositionalEncoding2D，默认src通道为3）
class MultiHeadSelfAttentionWithSourcePE(nn.Module):
    """自注意力外层模块（基于ResidualCrossAttentionBlock实现，q=k=v）"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, use_position_encoding=True, num_blocks=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_position_encoding = use_position_encoding
        self.num_blocks = num_blocks
        
        if use_position_encoding:
            self.pos_encoding = RotaryPositionEmbedding2D(
                head_dim = (hidden_size*2)//num_heads
            )
        else:
            self.pos_encoding = None
        
        # 创建多个残差注意力块（复用原有ResidualCrossAttentionBlock，保持不变）
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = ResidualCrossAttentionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                q_channels=hidden_size * 2,
                k_channels=hidden_size * 2,
                v_channels=hidden_size * 2
            )
            
            # 自注意力：q/k/v共享同一个位置编码（保持不变）
            if use_position_encoding:
                block.set_position_encodings(
                    self.pos_encoding,  # q的位置编码
                    self.pos_encoding,  # k的位置编码（和q相同）
                    self.pos_encoding   # v的位置编码（和q相同）
                )
            
            self.blocks.append(block)
        
        # 最终的归一化和dropout（和crossattn模块保持一致，不变）
        self.final_norm = nn.LayerNorm(hidden_size * 2)
        self.final_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        自注意力前向传播：q=k=v=x（保持调用逻辑不变，适配主模型）
        参数：
        - x: 输入张量，形状 (batch_size, c, h, w)
        返回：
        - output: 输出张量（形状与x完全一致）
        - intermediate_outputs: 训练时返回中间块输出，测试时返回None
        """
        # 保存原始输入用于最终残差连接（保持不变）
        original_input = x.clone()
        
        # 存储中间特征（可选，保持不变）
        intermediate_outputs = []
        
        # 逐个处理残差块：自注意力核心逻辑（fq=fk=fv=x，保持不变）
        for i, block in enumerate(self.blocks):
            x = block(
                fq=x, fk=x, fv=x,  # 关键：q=k=v=x
                use_position_encoding=self.use_position_encoding
            )
            
            # 保存中间输出（如果需要，保持不变）
            if self.training:
                intermediate_outputs.append(x.clone())
        
        # 最终归一化（保持逻辑不变，确保维度对齐）
        batch_size, c, h, w = x.shape
        seq_len = h * w
        
        x_seq = x.reshape(batch_size, c, seq_len).permute(0, 2, 1)
        x_seq = self.final_norm(x_seq)
        x = x_seq.permute(0, 2, 1).reshape(batch_size, c, h, w)
        
        # 最终的全局残差连接（保持不变）
        x = x + original_input
        
        # 最终的dropout（保持不变）
        x = self.final_dropout(x)
        
        # 保持返回格式：适配主模型的 `curr_fvis, _ = self.self_attn_vis(...)` 调用
        return x, intermediate_outputs if self.training else None
