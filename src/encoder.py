import torch
import torch.nn as nn
from src.encoder_block import EncoderBlock
from src.embeddings import Embeddings, PositionalEncoding

class TransformerEncoder(nn.Module):
    """
    Full Transformer Encoder.
    
    Structure:
    Input IDs -> Embedding -> Positional Encoding -> N x EncoderBlock -> Output
    """
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 512, 
                 num_heads: int = 8, 
                 num_layers: int = 6, 
                 d_ff: int = 2048, 
                 max_len: int = 5000, 
                 dropout: float = 0.1):
        super().__init__()
        
        # 1) Embedding layer
        self.embedding = Embeddings(d_model, vocab_size)
        
        # 2) Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        
        # 3) Stack of N Encoder Blocks
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 4) Final Layer Norm (optional but recommended)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, return_attention=False):
        """
        x: (Batch, SeqLen) - Input token IDs
        mask: (Batch, 1, 1, SeqLen) - Optional mask (e.g., padding mask)
        return_attention: If True, returns a list of attention maps from each layer
        """
        # 1. Embed & Add Position Info
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        attentions = []
        
        # 2. Apply N Encoder Blocks
        for layer in self.layers:
            # Note: Our current EncoderBlock doesn't take mask yet, 
            # but we should prepare for it. 
            # For now, we'll just pass x.
            x, attn_weights = layer(x)
            if return_attention:
                attentions.append(attn_weights)
            
        # 3. Final Normalization
        x = self.norm(x)
        
        if return_attention:
            return x, attentions
        return x
