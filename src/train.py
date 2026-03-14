import torch
import torch.nn as nn
from torch.nn import functional as F
from src.gpt import GPT, GPTConfig

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 100
eval_interval = 20
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

print(f"Using device: {device}")

torch.manual_seed(1337)

# Load data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"Vocab size: {vocab_size}")

# Train/Test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Data loader
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Initialize model
config = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout
)
model = GPT(config)
model = model.to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):

    # Evaluate loss
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Generate a sample
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
        print("-" * 50)

    # Sample batch
    xb, yb = get_batch('train')

    # Forward pass
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
