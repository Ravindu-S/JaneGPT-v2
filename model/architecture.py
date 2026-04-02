"""
JaneGPT v2 Model Architecture

A lightweight decoder-only transformer with classification head
for intent classification. Features modern architecture components:
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA)
- SwiGLU feed-forward networks
- RMSNorm

Created by Ravindu Senanayake
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# Intent labels — exact order matters for classification
INTENT_LABELS = [
    "volume_up", "volume_down", "volume_set", "volume_mute",
    "brightness_up", "brightness_down", "brightness_set",
    "media_play", "media_pause", "media_next", "media_previous",
    "browser_search", "app_launch", "app_close", "app_switch",
    "set_reminder", "screenshot", "read_screen", "explain_screen",
    "undo", "chat", "quit_jane",
]
INTENT_TO_ID = {label: i for i, label in enumerate(INTENT_LABELS)}
ID_TO_INTENT = {i: label for i, label in enumerate(INTENT_LABELS)}
NUM_INTENTS = len(INTENT_LABELS)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)."""
    def __init__(self, head_dim, max_seq_len=512, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos_cached', torch.cos(freqs))
        self.register_buffer('sin_cached', torch.sin(freqs))

    def forward(self, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rope(x, cos, sin):
    """Apply rotary position embeddings to input tensor."""
    head_dim = x.shape[-1]
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2)
    sin = sin.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2)
    return x * cos + rotated * sin


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).
    
    Uses fewer KV heads than query heads for memory efficiency
    while maintaining attention quality.
    """
    def __init__(self, embed_dim, num_heads, num_kv_heads, head_dim,
                 max_seq_len, dropout, rope_theta):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(head_dim, max_seq_len, rope_theta)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(seq_len)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out)


class SwiGLUFeedForward(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(self, embed_dim, ff_hidden, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, ff_hidden, bias=False)
        self.w2 = nn.Linear(ff_hidden, embed_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, ff_hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Single transformer block with GQA and SwiGLU."""
    def __init__(self, embed_dim, num_heads, num_kv_heads, head_dim,
                 ff_hidden, max_seq_len, dropout, rope_theta):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.attn = GroupedQueryAttention(
            embed_dim, num_heads, num_kv_heads, head_dim,
            max_seq_len, dropout, rope_theta
        )
        self.ff = SwiGLUFeedForward(embed_dim, ff_hidden, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class JaneGPTv2Classifier(nn.Module):
    """
    JaneGPT v2 Intent Classifier.
    
    A decoder-only transformer with a classification head
    for 22-class intent classification.
    
    Args:
        vocab_size: Vocabulary size (default: 8192)
        embed_dim: Embedding dimension (default: 256)
        num_heads: Number of attention heads (default: 8)
        num_kv_heads: Number of KV heads for GQA (default: 4)
        num_layers: Number of transformer layers (default: 8)
        ff_hidden: Feed-forward hidden dimension (default: 672)
        max_seq_len: Maximum sequence length (default: 256)
        dropout: Dropout rate (default: 0.1)
        rope_theta: RoPE theta parameter (default: 10000.0)
    """
    def __init__(self, vocab_size=8192, embed_dim=256, num_heads=8,
                 num_kv_heads=4, num_layers=8, ff_hidden=672,
                 max_seq_len=256, dropout=0.1, rope_theta=10000.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        head_dim = embed_dim // num_heads

        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, num_kv_heads, head_dim,
                ff_hidden, max_seq_len, dropout, rope_theta
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.intent_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, NUM_INTENTS),
        )

    def forward(self, x, labels=None):
        x = self.dropout(self.token_embedding(x))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        pooled = x[:, -1, :]
        logits = self.intent_head(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return logits, loss

    @torch.no_grad()
    def predict(self, x):
        logits, _ = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        confidence, predicted = torch.max(probs, dim=-1)
        return predicted.item(), confidence.item()