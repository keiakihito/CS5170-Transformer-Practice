import math
import torch
import torch.nn as nn
from src.embeddings import Embeddings, PositionalEncoding

def test_embeddings_shape():
    """
    Check if Embeddings returns correct shape.
    """
    B, T = 2, 10
    d_model = 16
    vocab_size = 100
    
    emb = Embeddings(d_model=d_model, vocab_size=vocab_size)
    x = torch.randint(0, vocab_size, (B, T))
    
    out = emb(x)
    assert out.shape == (B, T, d_model)

def test_positional_encoding_determinism():
    """
    Check if PE is deterministic (fixed sin/cos waves).
    """
    d_model = 16
    max_len = 50
    pe = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
    
    # Create dummy input of zeros
    x = torch.zeros(1, max_len, d_model)
    
    # Forward pass 1
    out1 = pe(x)
    
    # Forward pass 2
    out2 = pe(x)
    
    assert torch.allclose(out1, out2)
    
    # Check values at pos=0 (should be sin(0)=0 for even dims, cos(0)=1 for odd dims)
    # pe[0, 0, 0] -> sin(0) = 0
    # pe[0, 0, 1] -> cos(0) = 1
    assert torch.isclose(out1[0, 0, 0], torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(out1[0, 0, 1], torch.tensor(1.0), atol=1e-5)

def test_position_difference():
    """
    Check if different positions get different encodings.
    """
    d_model = 16
    pe = PositionalEncoding(d_model=d_model, max_len=10, dropout=0.0)
    x = torch.zeros(1, 10, d_model)
    out = pe(x)
    
    # Pos 0 and Pos 1 should be different
    pos0 = out[0, 0, :]
    pos1 = out[0, 1, :]
    
    assert not torch.allclose(pos0, pos1)
