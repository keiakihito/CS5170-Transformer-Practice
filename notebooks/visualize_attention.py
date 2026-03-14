import torch
import matplotlib.pyplot as plt
import seaborn as sns
from src.encoder import TransformerEncoder
import os

def visualize_attention():
    # 1. Setup Model
    vocab_size = 100
    d_model = 32
    num_heads = 4
    num_layers = 2
    
    # Deterministic behavior
    torch.manual_seed(42)
    
    encoder = TransformerEncoder(
        vocab_size=vocab_size, 
        d_model=d_model, 
        num_heads=num_heads, 
        num_layers=num_layers
    )
    encoder.eval()
    
    # 2. Prepare Input
    # Let's imagine a sentence with 5 words: [A, B, C, D, E]
    T = 5
    x = torch.randint(0, vocab_size, (1, T))
    
    # 3. Forward Pass with Attention Return
    with torch.no_grad():
        out, attentions = encoder(x, return_attention=True)
    
    # attentions is a list of tensors, one per layer
    # shape: (B, num_heads, T, T)
    
    # 4. Plot
    # We will plot Layer 0 and Layer 1
    # Each layer has 4 heads.
    
    # Create a grid of subplots: Rows=Layers, Cols=Heads
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(15, 8))
    
    # Token labels (just indices for now)
    tokens = [f"Token {i}" for i in range(T)]
    
    for layer_idx, attn_map in enumerate(attentions):
        # attn_map: (1, 4, 5, 5) -> take first batch -> (4, 5, 5)
        attn_map = attn_map[0]
        
        for head_idx in range(num_heads):
            ax = axes[layer_idx, head_idx]
            
            # Heatmap
            sns.heatmap(
                attn_map[head_idx], 
                ax=ax, 
                cmap="viridis", 
                vmin=0, vmax=1,
                xticklabels=tokens,
                yticklabels=tokens,
                cbar=False,
                annot=True, # Show values
                fmt=".2f"
            )
            
            ax.set_title(f"Layer {layer_idx} - Head {head_idx}")
            
    plt.tight_layout()
    output_path = "attention_map.png"
    plt.savefig(output_path)
    print(f"Saved attention map to {output_path}")

if __name__ == "__main__":
    visualize_attention()
