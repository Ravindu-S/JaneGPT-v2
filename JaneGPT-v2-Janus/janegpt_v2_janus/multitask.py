import torch
import torch.nn as nn
import torch.nn.functional as F

from .architecture import JaneGPTBackbone


def pool_last_nonpad(hidden, attention_mask=None):
    """
    hidden: (B,T,D)
    attention_mask: (B,T) 1=real token, 0=pad
    returns pooled: (B,D)
    """
    if attention_mask is None:
        return hidden[:, -1, :]

    lengths = attention_mask.sum(dim=1).clamp(min=1)  # (B,)
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, hidden.size(-1))  # (B,1,D)
    pooled = hidden.gather(1, idx).squeeze(1)  # (B,D)
    return pooled


class JaneGPTv3MultiTask(nn.Module):
    """
    Multi-task model:
    - domain classification (pooled)
    - action classification (pooled)
    - slot BIO tagging (per token)
    """
    def __init__(self, vocab_size=8192, embed_dim=256, num_heads=8, num_kv_heads=4,
                 num_layers=8, ff_hidden=672, max_seq_len=256,
                 dropout=0.1, rope_theta=10000.0,
                 num_domains=10, num_actions=33, num_slot_labels=1):
        super().__init__()

        self.backbone = JaneGPTBackbone(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            ff_hidden=ff_hidden,
            max_seq_len=max_seq_len,
            dropout=dropout,
            rope_theta=rope_theta,
        )

        self.dropout = nn.Dropout(dropout)

        self.domain_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_domains),
        )
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_actions),
        )

        self.slot_head = nn.Linear(embed_dim, num_slot_labels)

    def forward(self, input_ids, attention_mask=None,
                labels_domain=None, labels_action=None, labels_slots=None,
                causal=False):
        hidden = self.backbone(input_ids, attention_mask=attention_mask, causal=causal)  # (B,T,D)
        hidden = self.dropout(hidden)

        pooled = pool_last_nonpad(hidden, attention_mask=attention_mask)  # (B,D)

        logits_domain = self.domain_head(pooled)         # (B, num_domains)
        logits_action = self.action_head(pooled)         # (B, num_actions)
        logits_slots  = self.slot_head(hidden)           # (B, T, num_slot_labels)

        out = {
            "logits_domain": logits_domain,
            "logits_action": logits_action,
            "logits_slots": logits_slots,
        }

        if labels_domain is not None and labels_action is not None and labels_slots is not None:
            loss_domain = F.cross_entropy(logits_domain, labels_domain)
            loss_action = F.cross_entropy(logits_action, labels_action)

            # labels_slots should use -100 for padding positions
            loss_slots = F.cross_entropy(
                logits_slots.view(-1, logits_slots.size(-1)),
                labels_slots.view(-1),
                ignore_index=-100
            )

            # slot tagging is often harder; give it more weight
            loss = 1.0 * loss_domain + 1.0 * loss_action + 1.5 * loss_slots

            out.update({
                "loss": loss,
                "loss_domain": loss_domain,
                "loss_action": loss_action,
                "loss_slots": loss_slots,
            })

        return out
