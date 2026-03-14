import torch
import pytest
from src.multi_head_attention import MultiHeadAttention

def test_mha_output_shape():
    """
    Check if MultiHeadAttention returns the correct shape (B, T, D).
    """
    B, T, D = 2, 10, 16
    num_heads = 4
    
    mha = MultiHeadAttention(d_model=D, num_heads=num_heads)
    x = torch.randn(B, T, D)
    
    out, weights = mha(x)
    
    assert out.shape == (B, T, D)
    # weights: (B, num_heads, T, T)
    assert weights.shape == (B, num_heads, T, T)

def test_mha_causal_mask():
    """
    Check if causal masking works in MultiHeadAttention.
    """
    B, T, D = 1, 4, 8
    num_heads = 2
    
    # Enable causal masking
    mha = MultiHeadAttention(d_model=D, num_heads=num_heads, causal=True)
    
    x = torch.randn(B, T, D)
    x.requires_grad = True
    
    out, _ = mha(x)
    
    # Select output at t=0 (first token)
    out_0 = out[:, 0, :]
    
    # Compute gradients of out_0 w.r.t input x
    loss = out_0.sum()
    loss.backward()
    
    # Gradient at x[:, 1:, :] should be 0 because t=0 cannot attend to t=1,2,3
    grad_future = x.grad[:, 1:, :]
    
    assert torch.allclose(grad_future, torch.zeros_like(grad_future), atol=1e-5)

def test_head_diversity():
    """
    Check if different heads produce different attention patterns.
    """
    B, T, D = 1, 4, 16
    num_heads = 4 # 4 heads
    
    mha = MultiHeadAttention(d_model=D, num_heads=num_heads)
    
    # We want to peek at the internal attention weights.
    # To do this cleanly, we'll manually invoke the internal logic 
    # or monkey-patch the MiniAttention forward? 
    # Or we can just use the fact that we can call internal modules.
    
    x = torch.randn(B, T, D)
    
    # Run forward pass steps manually to inspect weights
    qkv = mha.c_attn(x)
    q, k, v = qkv.chunk(3, dim=2)
    
    q = q.view(B, T, num_heads, mha.d_k).transpose(1, 2)
    k = k.view(B, T, num_heads, mha.d_k).transpose(1, 2)
    v = v.view(B, T, num_heads, mha.d_k).transpose(1, 2)
    
    q = q.reshape(B * num_heads, T, mha.d_k)
    k = k.reshape(B * num_heads, T, mha.d_k)
    v = v.reshape(B * num_heads, T, mha.d_k)
    
    # Call internal attention
    _, _, weights = mha.attention(q, k, v)
    
    # weights shape: (B*num_heads, T, T)
    # Reshape to separate heads: (B, num_heads, T, T)
    weights = weights.view(B, num_heads, T, T)
    
    # Check if head 0 and head 1 are different
    # Since weights are randomly initialized, they should be different.
    head0 = weights[0, 0]
    head1 = weights[0, 1]
    
    assert not torch.allclose(head0, head1)
