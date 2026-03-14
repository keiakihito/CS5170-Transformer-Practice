import math
import torch
import torch.nn as nn

class Embeddings(nn.Module):
    """
    Standard Lookup Table + Scaling
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # Scale embedding by sqrt(d_model) as per paper
        # This scaling helps balance the variance of the embeddings with the positional encoding
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension d_model
    as the embeddings, so that the two can be summed.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        
        # Position indices: column vector [0, 1, 2, ..., max_len-1] -> (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # Frequency term: 1 / (10000^(2i/d_model))
        # Calculated in log space for numerical stability: exp( -log(10000) * (2i / d_model) )
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        # Apply sin to even indices (0, 2, 4...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices (1, 3, 5...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_len, d_model) for broadcasting
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (Batch, SeqLen, Dim)
        """
        # Add positional encoding to input
        # We slice self.pe to the sequence length of x
        # x + PE(pos)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
