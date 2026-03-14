import torch
import torch.nn as nn
import pytest
from src.encoder_block import EncoderBlock

def test_encoder_block_shape():
    """
    Check if EncoderBlock preserves shape.
    """
    B, T, D = 2, 10, 16
    num_heads = 4
    
    block = EncoderBlock(d_model=D, num_heads=num_heads)
    x = torch.randn(B, T, D)
    
    out, _ = block(x)
    
    assert out.shape == (B, T, D)

def test_residual_connection():
    """
    Verify that the residual connection is working.
    If we zero out the sublayer weights, the output should closely resemble the input (after norm).
    """
    B, T, D = 1, 4, 8
    num_heads = 2
    
    block = EncoderBlock(d_model=D, num_heads=num_heads, dropout=0.0)
    
    # Initialize weights to near-zero so sublayers output ~0
    # Then output = Norm(x + 0) = Norm(x)
    for p in block.parameters():
        nn.init.constant_(p, 0.0)
    
    # But LayerNorm has learnable parameters (gamma, beta).
    # We need to set gamma=1, beta=0 to make it identity-like.
    nn.init.ones_(block.norm1.weight)
    nn.init.zeros_(block.norm1.bias)
    nn.init.ones_(block.norm2.weight)
    nn.init.zeros_(block.norm2.bias)
        
    x = torch.randn(B, T, D)
    
    # Since sublayers output 0, the block should output LayerNorm(LayerNorm(x)).
    # Let's just check if it's NOT zero, meaning x passed through.
    out, _ = block(x)
    
    assert not torch.allclose(out, torch.zeros_like(out))
    
    # More specifically, if sublayers are 0, output is normalized x.
    # Let's compare out with manually normalized x.
    # Note: LayerNorm is applied twice.
    ln = nn.LayerNorm(D)
    ln.weight.data.fill_(1.0)
    ln.bias.data.fill_(0.0)
    expected = ln(ln(x))
    
    assert torch.allclose(out, expected, atol=1e-5)

def test_layer_norm_statistics():
    """
    Check if LayerNorm normalizes the output statistics.
    Output should have mean ~0 and std ~1 over the last dimension.
    """
    B, T, D = 4, 10, 32
    num_heads = 4
    
    block = EncoderBlock(d_model=D, num_heads=num_heads)
    x = torch.randn(B, T, D) * 10 + 5  # Input with mean 5, std 10
    
    out, _ = block(x)
    
    # Check mean and std across the feature dimension (D)
    # They should be close to 0 and 1 respectively.
    # Note: LayerNorm applies per token.
    mean = out.mean(dim=-1)
    std = out.std(dim=-1, unbiased=False)
    
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-1) # Relax tolerance a bit
    assert torch.allclose(std, torch.ones_like(std), atol=1e-1)

