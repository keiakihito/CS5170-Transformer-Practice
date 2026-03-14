import torch
import torch.nn as nn
from src.multi_head_attention import MultiHeadAttention

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    A simple 2-layer MLP applied to each position independently.
    
    Structure:
    Input -> Linear(4*D) -> ReLU -> Linear(D) -> Output
    """
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        # Usually d_ff is 4 * d_model
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class EncoderBlock(nn.Module):
    """
    Standard Transformer Encoder Block (Post-LN style).
    
    Structure:
    x -> MultiHead -> Add -> LayerNorm -> FFN -> Add -> LayerNorm
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        
        # 1) Multi-Head Attention sublayer
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 2) Feed-Forward sublayer
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        # Sublayer 1: Self-Attention
        # Residual connection: x + Sublayer(x)
        # Post-LN: LayerNorm(x + Sublayer(x))
        attn_out, weights = self.self_attn(x)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # Sublayer 2: Feed-Forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, weights
