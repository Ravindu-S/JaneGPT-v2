import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=512, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", torch.cos(freqs))
        self.register_buffer("sin_cached", torch.sin(freqs))

    def forward(self, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rope(x, cos, sin):
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)

    # broadcast to (1,1,T,head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2)
    sin = sin.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 2)

    return x * cos + rotated * sin


class GroupedQueryAttention(nn.Module):
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

    def forward(self, x, attention_mask=None, causal=False):
        """
        x: (B,T,D)
        attention_mask: (B,T) where 1=real token, 0=pad
        causal: if True, apply future mask (decoder-style)
        """
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

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,T,T)

        # Key padding mask: prevent attending TO pad tokens
        if attention_mask is not None:
            # mask shape -> (B,1,1,T)
            key_pad = (attention_mask == 0)[:, None, None, :]
            scores = scores.masked_fill(key_pad, float("-inf"))

        # Optional causal mask (OFF for v3 slot tagging / accuracy)
        if causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_hidden, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, ff_hidden, bias=False)
        self.w2 = nn.Linear(ff_hidden, embed_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, ff_hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
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

    def forward(self, x, attention_mask=None, causal=False):
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask, causal=causal)
        x = x + self.ff(self.norm2(x))
        return x


class JaneGPTBackbone(nn.Module):
    """
    v3 Backbone: returns hidden states (B,T,D) and supports attention_mask + causal flag.
    IMPORTANT: module/param names match v2 (token_embedding, layers.*, norm, dropout)
    so we can warm-start from v2 weights.
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

    def forward(self, input_ids, attention_mask=None, causal=False):
        x = self.dropout(self.token_embedding(input_ids))
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, causal=causal)
        x = self.norm(x)
        return x
