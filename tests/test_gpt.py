import torch
import pytest
from src.gpt import GPT, GPTConfig

def test_gpt_forward():
    """
    Check if GPT forward pass works.
    """
    config = GPTConfig(
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_embd=32,
        block_size=10
    )
    model = GPT(config)
    
    B, T = 2, 5
    idx = torch.randint(0, config.vocab_size, (B, T))
    
    logits, loss = model(idx)
    
    assert logits.shape == (B, T, config.vocab_size)
    assert loss is None

def test_gpt_loss():
    """
    Check if GPT calculates loss correctly.
    """
    config = GPTConfig(
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_embd=32,
        block_size=10
    )
    model = GPT(config)
    
    B, T = 2, 5
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    
    logits, loss = model(idx, targets)
    
    assert loss is not None
    assert loss.item() > 0

def test_gpt_generate():
    """
    Check if GPT generation works.
    """
    config = GPTConfig(
        vocab_size=100,
        n_layer=2,
        n_head=4,
        n_embd=32,
        block_size=10
    )
    model = GPT(config)
    
    B, T = 1, 3
    idx = torch.randint(0, config.vocab_size, (B, T))
    
    # Generate 5 new tokens
    new_tokens = 5
    out = model.generate(idx, max_new_tokens=new_tokens)
    
    assert out.shape == (B, T + new_tokens)
