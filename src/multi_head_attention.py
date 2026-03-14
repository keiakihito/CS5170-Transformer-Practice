import torch
import torch.nn as nn
from src.mini_attention import MiniAttention

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.
    
    Structure:
    Input -> Linear(3*D) -> Split Heads -> Scaled Dot-Product Attention -> Merge Heads -> Linear(D) -> Output
    """
    def __init__(self, d_model: int, num_heads: int, causal: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # 1) Linear projection for Q, K, V
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        
        # 2) Underlying attention mechanism
        self.attention = MiniAttention(causal=causal)
        
        # 3) Output projection
        self.c_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x):
        """
        Reshape input to parallelize over heads.
        Input:  (B, T, D)
        Output: (B * num_heads, T, d_k)
        """
        B, T, D = x.size()
        # (B, T, D) -> (B, T, num_heads, d_k) -> (B, num_heads, T, d_k)
        x = x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        # Collapse batch and heads: (B * num_heads, T, d_k)
        return x.reshape(B * self.num_heads, T, self.d_k)

    def _merge_heads(self, x, B, T):
        """
        Reverse of _split_heads.
        Input:  (B * num_heads, T, d_k)
        Output: (B, T, D)
        """
        # Unfold batch: (B, num_heads, T, d_k)
        x = x.view(B, self.num_heads, T, self.d_k)
        # Transpose back: (B, T, num_heads, d_k)
        x = x.transpose(1, 2)
        # Concatenate heads: (B, T, D)
        return x.contiguous().view(B, T, self.d_model)

    def forward(self, x):
        """
        x: (B, T, D) - Batch, Time, Dimension (d_model)
        """
        B, T, D = x.size()
        
        # 1) Linear Projection & Split Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=2)
        
        # 2) Split Heads
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        
        # 3) Interaction (Scaled Dot-Product Attention)
        attn_out, scores, weights = self.attention(q, k, v)
        # weights shape: (B*num_heads, T, T)
        
        # 4) Merge Heads
        attn_out = self._merge_heads(attn_out, B, T)
        
        # 5) Output Projection
        out = self.c_proj(attn_out)
        
        # Unfold weights for visualization: (B, num_heads, T, T)
        weights = weights.view(B, self.num_heads, T, T)
        
        return out, weights
