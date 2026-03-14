import torch
import pytest
from src.mini_attention import MiniAttention

def test_output_shape():
    """
    Check if the output tensor has the correct shape (B, T, D).
    """
    B, T, D = 2, 4, 8
    attention = MiniAttention()
    Q = torch.randn(B, T, D)
    K = torch.randn(B, T, D)
    V = torch.randn(B, T, D)
    
    out, scores, weights = attention(Q, K, V)
    
    assert out.shape == (B, T, D)
    assert scores.shape == (B, T, T)
    assert weights.shape == (B, T, T)

def test_diagonal_dominance():
    """
    Check if attention weights are highest on the diagonal when Q == K.
    This simulates 'looking at itself' being the strongest signal.
    """
    B, T, D = 1, 4, 8
    attention = MiniAttention()
    
    # Create orthogonal vectors to ensure self-similarity is strictly highest
    # Q[0] will be like [[1,0,..], [0,1,..], ...]
    Q = torch.eye(T, D).unsqueeze(0) 
    K = Q.clone()
    V = Q.clone()
    
    out, scores, weights = attention(Q, K, V)
    
    # Check if diagonal elements are the largest in each row
    for i in range(T):
        row = weights[0, i, :]
        assert torch.argmax(row).item() == i
        
    # Also check that the weight is close to 1 (since other dot products are 0)
    # softmax([large, -inf, -inf...]) -> [1, 0, 0...]
    # Here dot products are 1 (diagonal) and 0 (off-diagonal).
    # softmax([1/sqrt(D), 0, 0...]) won't be exactly 1, but diagonal should be largest.
    
def test_uniform_attention():
    """
    Check if attention is uniform when all keys are identical.
    """
    B, T, D = 1, 4, 8
    attention = MiniAttention()
    Q = torch.randn(B, T, D)
    
    # Make K identical for all time steps
    # If all keys are the same, the query will match them all equally
    k_vec = torch.randn(1, 1, D)
    K = k_vec.expand(B, T, D)
    V = torch.randn(B, T, D)
    
    out, scores, weights = attention(Q, K, V)
    
    # Each row should be roughly 1/T
    expected_weight = 1.0 / T
    assert torch.allclose(weights, torch.full_like(weights, expected_weight), atol=1e-5)

def test_attention_sum_to_one():
    """
    Check if attention weights sum to 1 for each row.
    """
    B, T, D = 2, 4, 8
    attention = MiniAttention()
    Q = torch.randn(B, T, D)
    K = torch.randn(B, T, D)
    V = torch.randn(B, T, D)
    
    out, scores, weights = attention(Q, K, V)
    
    # Sum over the last dimension (key dimension) should be 1
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

def test_causal_mask():
    """
    Check if causal mask prevents attending to future tokens.
    """
    B, T, D = 1, 4, 8
    attention = MiniAttention(causal=True)
    Q = torch.randn(B, T, D)
    K = torch.randn(B, T, D)
    V = torch.randn(B, T, D)
    
    out, scores, weights = attention(Q, K, V)
    
    # Upper triangle (excluding diagonal) should be 0
    # weights shape: (B, T, T)
    upper_tri = torch.triu(weights[0], diagonal=1)
    assert torch.allclose(upper_tri, torch.zeros_like(upper_tri), atol=1e-5)
    
    # Also check that rows sum to 1 even with masking
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
