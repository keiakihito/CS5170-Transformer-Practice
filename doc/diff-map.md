# 🗺️ Transformer Evolution Diff Map

Original Transformer (Attention Is All You Need / GPT-2) と Modern LLM (Llama 3 / Mistral) の技術的差分マップ。

| Component | Original (minGPT) | Modern (Llama 3) | Purpose & Benefits |
| :--- | :--- | :--- | :--- |
| **Normalization** | `LayerNorm` | `RMSNorm` | **Efficiency**<br>平均(Mean)を引く処理を省略し、二乗平均平方根(RMS)のみで正規化。計算コスト削減と学習安定化。 |
| **Activation** | `ReLU` / `GELU` | `SwiGLU` | **Expressivity**<br>Gated Linear Unit (GLU) + Swish関数。パラメータ数は増えるが(Linear 3層)、学習効率と精度が向上。 |
| **Position Embedding** | `Learned Positional Embedding`<br>(Absolute) | `RoPE`<br>(Rotary Positional Embedding) | **Extrapolation**<br>絶対位置を加算するのではなく、Query/Keyベクトルを回転させることで相対位置を表現。長文対応力が高い。 |
| **Attention** | `Standard Attention`<br>(`softmax(QK^T)V`) | `Grouped-Query Attention (GQA)`<br>`FlashAttention` | **Speed & Memory**<br>GQA: KVキャッシュのサイズを削減。<br>FlashAttention: GPUメモリI/Oを最適化し、数倍の高速化。 |
| **Bias Terms** | `Linear(bias=True)` | `Linear(bias=False)` | **Simplicity**<br>多くの層でバイアス項を削除。LayerNorm/RMSNormが中心化を行うため不要とされることが多い。 |

---

## 1. Normalization: LayerNorm vs RMSNorm

### LayerNorm (Original)
```python
# pytorch default
x = (x - x.mean()) / x.std() * weight + bias
```

### RMSNorm (Modern)
```python
# Root Mean Square Norm
# No mean subtraction, no bias term usually
x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
```

---

## 2. Activation: GELU vs SwiGLU

### GELU MLP (Original)
構造: `Linear -> GELU -> Linear`

```python
class MLP(nn.Module):
    def __init__(self):
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj  = nn.Linear(4 * n_embd, n_embd)
        self.act     = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))
```

### SwiGLU MLP (Modern)
構造: `(Linear_Gate * Linear_Val) -> Linear_Out`
※ 隠れ層の次元は通常 `4 * n_embd` から `8/3 * n_embd` 程度に調整される。

```python
class SwiGLU(nn.Module):
    def __init__(self):
        self.w1 = nn.Linear(n_embd, n_hidden, bias=False) # Gate
        self.w2 = nn.Linear(n_embd, n_hidden, bias=False) # Value
        self.w3 = nn.Linear(n_hidden, n_embd, bias=False) # Output

    def forward(self, x):
        # SiLU (Swish) * Value
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
```

---

## 3. Position Embedding: Absolute vs RoPE

### Learned Absolute PE (Original)
入力に足し算するだけ。

```python
# Input Embedding
x = tok_emb + pos_emb
```

### Rotary Positional Embedding (Modern)
Attention計算の直前に、QueryとKeyを回転させる。

```python
# Inside Attention Forward
q, k = apply_rotary_emb(q, k, freqs_cis)
# Then attention...
att = (q @ k.transpose(...)) ...
```

---

## 4. Attention: MHA vs GQA

### Multi-Head Attention (MHA)
Query, Key, Value のヘッド数が同じ。
`n_q_heads = n_kv_heads`

### Grouped-Query Attention (GQA)
Key/Value のヘッド数を減らす（グループ化）。
`n_q_heads = 32`, `n_kv_heads = 8` (例: Llama 2 70B)
これにより、推論時のKVキャッシュメモリ量が `1/4` になる。

```python
# GQA Concept
# Repeat KV heads to match Q heads count before attention
k = k.repeat_interleave(n_q_heads // n_kv_heads, dim=1)
v = v.repeat_interleave(n_q_heads // n_kv_heads, dim=1)
```
