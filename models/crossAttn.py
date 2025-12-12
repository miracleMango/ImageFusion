from models.SourcePositionalEncoding import SourcePositionalEncoding2D
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualCrossAttentionBlock(nn.Module):
    """通用的残差交叉注意力块"""
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, 
                 q_channels=None, k_channels=None, v_channels=None):
        """
        参数说明：
        - q_channels: query输入的通道数（如果为None，则使用hidden_size*2）
        - k_channels: key输入的通道数（如果为None，则使用hidden_size*2）
        - v_channels: value输入的通道数（如果为None，则使用hidden_size*2）
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = (hidden_size * 2) // num_heads
        self.q_channels = q_channels if q_channels is not None else hidden_size * 2
        self.k_channels = k_channels if k_channels is not None else hidden_size * 2
        self.v_channels = v_channels if v_channels is not None else hidden_size * 2

        assert self.head_dim * num_heads == hidden_size * 2, "hidden_size必须是num_heads的整数倍"

        # 投影层
        self.w_q = nn.Linear(self.q_channels, hidden_size * 2, bias=True)
        self.w_k = nn.Linear(self.k_channels, hidden_size * 2, bias=True)
        self.w_v = nn.Linear(self.v_channels, hidden_size * 2, bias=True)
        self.proj = nn.Linear(hidden_size * 2, hidden_size * 2, bias=True)
        
        # 位置编码（在外部传入，这里只定义但不创建）
        self.pos_encoding_q = None
        self.pos_encoding_k = None
        self.pos_encoding_v = None
        
        # Dropout和归一化
        self.dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_size * 2)  # 注意力后的归一化
        self.norm2 = nn.LayerNorm(hidden_size * 2)  # 前馈后的归一化
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.Dropout(dropout)
        )
        
        # 可学习的残差权重
        self.residual_weight = nn.Parameter(torch.tensor(1.0))
        
        self.register_buffer('scale', torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
    
    def set_position_encodings(self, pos_encoding_q, pos_encoding_k, pos_encoding_v):
        """设置位置编码（由外部传入）"""
        self.pos_encoding_q = pos_encoding_q
        self.pos_encoding_k = pos_encoding_k
        self.pos_encoding_v = pos_encoding_v
    
    def forward(self, fq, fk, fv, source_img_q, source_img_k, source_img_v, use_position_encoding=True):
        batch_size, c_q, h, w = fq.shape
        seq_len = h * w
        
        # 保存原始fq用于残差连接
        residual = fq.clone()
        
        # 步骤1：添加位置编码（如果启用且位置编码存在）
        if use_position_encoding and self.pos_encoding_q is not None:
            fq = self.pos_encoding_q(fq, source_img_q)
            fk = self.pos_encoding_k(fk, source_img_k)
            fv = self.pos_encoding_v(fv, source_img_v)
        
        # 将特征重塑为序列
        fq_seq = fq.reshape(batch_size, c_q, seq_len).permute(0, 2, 1)
        fk_seq = fk.reshape(batch_size, self.k_channels, seq_len).permute(0, 2, 1)
        fv_seq = fv.reshape(batch_size, self.v_channels, seq_len).permute(0, 2, 1)
        
        # 投影
        q = self.w_q(fq_seq)
        k = self.w_k(fk_seq)
        v = self.w_v(fv_seq)
        
        # 多头处理
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        
        # 注意力计算
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
            scale=self.scale
        )
        
        # 重组输出
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size * 2)
        output_seq = self.proj(attn_output)
        
        # 第一个残差连接（注意力输出 + 原始输入）
        output_seq = output_seq + self.residual_weight * self.residual_dropout(
            residual.reshape(batch_size, c_q, seq_len).permute(0, 2, 1)
        )
        
        # 层归一化
        output_seq = self.norm1(output_seq)
        
        # 前馈网络
        ffn_residual = output_seq.clone()
        ffn_output = self.ffn(output_seq)
        
        # 第二个残差连接（前馈输出 + 注意力输出）
        output_seq = ffn_residual + ffn_output
        
        # 最终层归一化
        output_seq = self.norm2(output_seq)
        
        # 转换回4D
        output = output_seq.permute(0, 2, 1).reshape(batch_size, c_q, h, w)
        
        return output


class MultiHeadCrossAttentionWithSourcePE(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, use_position_encoding=True,
                 src_q_channels=1, src_k_channels=1, src_v_channels=1, num_blocks=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_position_encoding = use_position_encoding
        self.num_blocks = num_blocks
        
        # 位置编码（所有块共享）
        if use_position_encoding:
            self.pos_encoding_q = SourcePositionalEncoding2D(src_q_channels, hidden_size * 2)
            self.pos_encoding_k = SourcePositionalEncoding2D(src_k_channels, hidden_size * 2)
            self.pos_encoding_v = SourcePositionalEncoding2D(src_v_channels, hidden_size * 2)
        else:
            self.pos_encoding_q = None
            self.pos_encoding_k = None
            self.pos_encoding_v = None
        
        # 创建多个残差注意力块
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
            
            # 设置位置编码（所有块共享相同的位置编码）
            if use_position_encoding:
                block.set_position_encodings(
                    self.pos_encoding_q,
                    self.pos_encoding_k,
                    self.pos_encoding_v
                )
            
            self.blocks.append(block)
        
        # 最终的归一化和dropout
        self.final_norm = nn.LayerNorm(hidden_size * 2)
        self.final_dropout = nn.Dropout(dropout)
        
    def forward(self, fq, fk, fv, source_img_q, source_img_k, source_img_v):
        # 保存原始输入用于最终残差连接
        original_input = fq.clone()
        
        # 存储中间特征（可选）
        intermediate_outputs = []
        
        # 逐个处理残差块
        for i, block in enumerate(self.blocks):
            fq = block(
                fq, fk, fv, 
                source_img_q, source_img_k, source_img_v,
                use_position_encoding=self.use_position_encoding
            )
            
            # 保存中间输出（如果需要）
            if self.training:
                intermediate_outputs.append(fq.clone())
        
        # 最终归一化
        batch_size, c, h, w = fq.shape
        seq_len = h * w
        
        fq_seq = fq.reshape(batch_size, c, seq_len).permute(0, 2, 1)
        fq_seq = self.final_norm(fq_seq)
        fq = fq_seq.permute(0, 2, 1).reshape(batch_size, c, h, w)
        
        # 最终的全局残差连接
        fq = fq + original_input
        
        # 最终的dropout
        fq = self.final_dropout(fq)
        
        return fq, intermediate_outputs if self.training else None


class MultiHeadCrossAttentionFusionWithSourcePE(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, use_position_encoding=True,
                 src_kv_channels=4, src_q_channels=3, num_blocks=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_position_encoding = use_position_encoding
        self.num_blocks = num_blocks
        
        # 位置编码（所有块共享）
        if use_position_encoding:
            self.pos_encoding_q = SourcePositionalEncoding2D(src_q_channels, hidden_size * 2)
            self.pos_encoding_k = SourcePositionalEncoding2D(src_kv_channels, hidden_size * 4)
            self.pos_encoding_v = SourcePositionalEncoding2D(src_kv_channels, hidden_size * 4)
        else:
            self.pos_encoding_q = None
            self.pos_encoding_k = None
            self.pos_encoding_v = None
        
        # 创建多个残差注意力块
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = ResidualCrossAttentionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                q_channels=hidden_size * 2,      # fq的通道数
                k_channels=hidden_size * 4,      # fk的通道数
                v_channels=hidden_size * 4       # fv的通道数
            )
            
            # 设置位置编码（所有块共享相同的位置编码）
            if use_position_encoding:
                block.set_position_encodings(
                    self.pos_encoding_q,
                    self.pos_encoding_k,
                    self.pos_encoding_v
                )
            
            self.blocks.append(block)
        
        # 最终的归一化和dropout
        self.final_norm = nn.LayerNorm(hidden_size * 2)
        self.final_dropout = nn.Dropout(dropout)
        
    def forward(self, fq, fk, fv, source_img_q, source_img_k, source_img_v):
        # 保存原始输入用于最终残差连接
        original_input = fq.clone()
        
        # 存储中间特征（可选）
        intermediate_outputs = []
        
        # 逐个处理残差块
        for i, block in enumerate(self.blocks):
            fq = block(
                fq, fk, fv, 
                source_img_q, source_img_k, source_img_v,
                use_position_encoding=self.use_position_encoding
            )
            
            # 保存中间输出（如果需要）
            if self.training:
                intermediate_outputs.append(fq.clone())
        
        # 最终归一化
        batch_size, c, h, w = fq.shape
        seq_len = h * w
        
        fq_seq = fq.reshape(batch_size, c, seq_len).permute(0, 2, 1)
        fq_seq = self.final_norm(fq_seq)
        fq = fq_seq.permute(0, 2, 1).reshape(batch_size, c, h, w)
        
        # 最终的全局残差连接
        fq = fq + original_input
        
        # 最终的dropout
        fq = self.final_dropout(fq)
        
        return fq, intermediate_outputs if self.training else None