import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniAttention(nn.Module):
    """
    Minimal scaled dot-product attention for learning.

    Tensor shape convention:

    B = batch size
        -> number of sentences processed in parallel
        example: B = 2 means 2 sentences at once

    T = sequence length (number of tokens per sentence)
        example: T = 4 means each sentence has 4 tokens

    D = hidden dimension (embedding size per token)
        example: D = 8 means each token is represented as an 8-dimensional vector

    So if B=2, T=4, D=8:

    Q.shape = (2, 4, 8)

        Q[0]        -> first sentence
        Q[0,1]      -> 2nd token of first sentence
        Q[0,1,:]    -> 8-dimensional vector for that token

    Same structure for K and V.
    """

    def __init__(self, causal: bool = False):
        super().__init__()
        self.causal = causal

    def forward(self, Q, K, V):
        """
        Q, K, V: (B, T, D)

        Example if B=2, T=4, D=8:

        Q = tensor of shape (2, 4, 8)

        Q[1,3] means:
            sentence index 1 (second sentence),
            token index 3 (fourth token),
            represented by an 8-dim vector.
        """

        B, T, D = Q.shape  # unpack shape

        # sanity check
        assert K.shape == (B, T, D)
        assert V.shape == (B, T, D)

        # 1) scaled dot-product scores: (B,T,D) @ (B,D,T) -> (B,T,T)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(D)

        # **2)**causal mask (prevent attending to future positions)
        if self.causal:
            # mask shape: (T,T) where True means "mask out"
            mask = torch.triu(torch.ones(T, T, device=Q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))

        # 3) softmax over keys dimension (last dim), distribution of weights total 1
        weights = F.softmax(scores, dim=-1)

        # 4) weighted sum of values: (B,T,T) @ (B,T,D) -> (B,T,D), linear combination of values
        out = weights @ V

        return out, scores, weights

"""
Note For 2)
Masking makes the model time-aware model.
In autoregressive language models (GPT / Llama style),
token i must NOT attend to tokens j > i (future tokens).
Example when T = 4:
torch.triu(torch.ones(4,4), diagonal=1) produces:
[[0, 1, 1, 1],
  [0, 0, 1, 1],
  [0, 0, 0, 1],
  [0, 0, 0, 0]]
Row = current token i
Column = candidate token j
1 (True) means: "mask out this future position"
So:
# token 0 can attend only to itself
# token 1 can attend to 0 and 1
# token 2 can attend to 0,1,2
# token 3 can attend to 0,1,2,3
We replace masked positions with -inf.
After softmax:
exp(-inf) = 0
So attention weight to future tokens becomes exactly 0.
This prevents information leakage during next-token prediction.
Without this mask, the model could "peek" at future words.
"""