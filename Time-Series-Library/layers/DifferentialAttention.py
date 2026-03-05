# --- layers/DifferentialAttention.py ---
# 该文件定义了适用于 iTransformer 的差分多头注意力机制
# Query, Key, Value 的形状为 [Batch, Variate, Head, Dim_per_Head]，注意力在 Variate 维度上进行。

import torch
import torch.nn as nn
import numpy as np
from math import sqrt


class DifferentialAttention(nn.Module):
    """
    差分注意力机制，专为 iTransformer 设计。
    Query, Key, Value 的形状为 [Batch, Variate, Head, Dim_per_Head]。
    注意力在 Variate 维度上进行，计算不同 Variate Token 之间的差异。
    """

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DifferentialAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        Forward pass for differential attention.
        Args:
            queries: [Batch, Variate, Head, Dim_per_Head] (注意：Variate 对应标准注意力的 L)
            keys: [Batch, Variate, Head, Dim_per_Head] (注意：Variate 对应标准注意力的 S)
            values: [Batch, Variate, Head, Dim_per_Head] (注意：Variate 对应标准注意力的 S)
            attn_mask: [Batch, Variate, Variate] (如果需要掩码)
            tau: Temperature parameter (unused here)
            delta: Delta parameter (unused here)
        Returns:
            out: [Batch, Variate, Head, Dim_per_Head]
            attn: [Batch, Head, Variate, Variate] if output_attention else None
        """
        B, L, H, E = queries.shape  # L is Variate, H is Head, E is Dim_per_Head
        _, S, _, D = values.shape   # S is Variate, D is Dim_per_Head

        scale = self.scale or 1. / sqrt(E)

        # queries: [B, L, H, E] -> [B, H, L, 1, E] -> [B, H, L, S, E]
        queries_expanded = queries.permute(0, 2, 1, 3).unsqueeze(-2).expand(-1, -1, -1, S, -1) # [B, H, L, S, E]
        # keys: [B, S, H, E] -> [B, H, 1, S, E] -> [B, H, L, S, E]
        keys_expanded = keys.permute(0, 2, 1, 3).unsqueeze(-3).expand(-1, -1, L, -1, -1)      # [B, H, L, S, E]

        # Calculate difference vector
        diff = queries_expanded - keys_expanded # [B, H, L, S, E]

        # Calculate squared L2 norm of the difference as similarity score
        scores_raw = torch.norm(diff, p=2, dim=-1) ** 2 # [B, H, L, S]

        # Apply negative sign (smaller diff means higher similarity)
        # Scale the scores
        scores = -scores_raw * scale # [B, H, L, S]

        # Apply attention mask if flag is set
        if self.mask_flag:
            if attn_mask is not None:
                # attn_mask: [B, L, S] -> [B, 1, L, S] for broadcasting over heads
                scores.masked_fill_(attn_mask.unsqueeze(1), -np.inf)

        # Compute softmax to get attention weights
        A = self.dropout(torch.softmax(scores, dim=-1)) # [B, H, L, S]

        # Aggregate values using attention weights
        # values: [B, S, H, D] -> [B, H, S, D]
        V_expanded = values.permute(0, 2, 1, 3) # [B, H, S, D]
        # A: [B, H, L, S], V_expanded: [B, H, S, D] -> context: [B, H, L, D]
        context = torch.matmul(A, V_expanded) # [B, H, L, D]

        # Convert back to original format [B, L, H, D]
        out = context.permute(0, 2, 1, 3) # [B, L, H, D]

        if self.output_attention:
            return out, A
        else:
            return out, None


class AttentionLayer(nn.Module):
    """
    封装差分注意力机制的层，与 iTransformer 的 EncoderLayer 兼容。
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape # L is Variate
        H = self.n_heads

        # Project and reshape to separate heads
        queries = self.query_projection(queries).view(B, L, H, -1) # [B, L, H, -1]
        keys = self.key_projection(keys).view(B, L, H, -1)       # [B, L, H, -1]
        values = self.value_projection(values).view(B, L, H, -1) # [B, L, H, -1]

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        # out is [B, L, H, -1], reshape back to [B, L, D]
        # 🔴 使用 reshape 替代 view 以防不连续问题
        out = out.reshape(B, L, -1) # [B, L, D]

        out = self.out_projection(out)

        return out, attn