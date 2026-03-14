import torch
import pytest
from src.encoder import TransformerEncoder

def test_encoder_output_shape():
    """
    Check if the full encoder outputs correct shape.
    """
    B, T = 2, 10
    d_model = 16
    vocab_size = 100
    num_layers = 2
    
    encoder = TransformerEncoder(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers)
    x = torch.randint(0, vocab_size, (B, T))
    
    out = encoder(x)
    
    assert out.shape == (B, T, d_model)

def test_contextualization():
    """
    Verify that the same word gets different representations depending on context.
    Example: 'bank' in 'river bank' vs 'bank account'.
    """
    vocab_size = 100
    d_model = 16
    encoder = TransformerEncoder(vocab_size=vocab_size, d_model=d_model, num_layers=2)
    
    # Let's say word ID 42 is 'bank'.
    # Context 1: [1, 42, 2] ('river bank .')
    seq1 = torch.tensor([[1, 42, 2]])
    
    # Context 2: [3, 42, 4] ('money bank !')
    seq2 = torch.tensor([[3, 42, 4]])
    
    # Forward pass
    # We turn off dropout for deterministic comparison
    encoder.eval()
    
    with torch.no_grad():
        out1 = encoder(seq1)
        out2 = encoder(seq2)
    
    # Extract the vector for 'bank' (index 1 in both sequences)
    bank_vec1 = out1[0, 1, :]
    bank_vec2 = out2[0, 1, :]
    
    # Initial embeddings would be identical (before PE),
    # but after Encoder layers, they should be different due to context.
    assert not torch.allclose(bank_vec1, bank_vec2, atol=1e-5)
    
    # Just to be sure, check if the distance is significant
    dist = torch.norm(bank_vec1 - bank_vec2)
    assert dist > 0.1

def test_position_sensitivity():
    """
    Check if swapping words changes the representation of other words.
    """
    vocab_size = 100
    d_model = 16
    encoder = TransformerEncoder(vocab_size=vocab_size, d_model=d_model, num_layers=2)
    encoder.eval()
    
    # Sequence: A B C
    seq1 = torch.tensor([[10, 20, 30]])
    
    # Sequence: C B A (swapped context around B)
    seq2 = torch.tensor([[30, 20, 10]])
    
    with torch.no_grad():
        out1 = encoder(seq1)
        out2 = encoder(seq2)
    
    # Check representation of 'B' (index 1)
    # Even though it's the same word at the same position index (1),
    # its neighbors changed, so its representation MUST change.
    vec_b1 = out1[0, 1, :]
    vec_b2 = out2[0, 1, :]
    
    assert not torch.allclose(vec_b1, vec_b2)
