import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    """
    Multi-head masked self-attention.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Key, Query, Value projections combined
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # Causal mask (Lower Triangular matrix of 1s)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        # Calculate query, key, values
        # (B, T, 3*C) -> (B, T, C) x 3
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        # (B, T, C) -> (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Scaled Dot-Product Attention
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply Causal Mask (prevent looking at future tokens)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Weighted sum of values
        y = att @ v 
        
        # Re-assemble heads
        # (B, nh, T, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    """
    Position-wise Feed-Forward Network.
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """ Transformer Block (Pre-LN) """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-LN architecture
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT Language Model """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Token & Position Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Final Layer Norm
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Output Head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialization
        self.apply(self._init_weights)
        
        # Scaling init for residual projections (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Report parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        if t > self.config.block_size:
            raise ValueError(f"Sequence length {t} exceeds block size {self.config.block_size}")
        
        # 1. Embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # (1, t)
        
        tok_emb = self.wte(idx) # (b, t, n_embd)
        pos_emb = self.wpe(pos) # (1, t, n_embd)
        
        x = self.drop(tok_emb + pos_emb)
        
        # 2. Transformer Blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        
        # 3. Output Logits
        logits = self.lm_head(x) # (b, t, vocab_size)

        # 4. Loss Calculation (Optional)
        loss = None
        if targets is not None:
            # Flatten to (Batch * Time, Vocab) for CrossEntropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive generation.
        """
        for _ in range(max_new_tokens):
            # Crop context if too long
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Focus on the last token's output
            logits = logits[:, -1, :] / temperature
            
            # Top-K Sampling (Optional)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Probability distribution
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
