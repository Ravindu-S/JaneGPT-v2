import json
import torch
from torch.utils.data import Dataset

from . import labels as L

PAD_ID = 0  # <pad>

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def spans_to_bio(offsets, spans):
    tags = ["O"] * len(offsets)
    spans = sorted(spans, key=lambda s: (s["start"], s["end"]))

    for sp in spans:
        stype = sp["type"]
        s0, s1 = int(sp["start"]), int(sp["end"])

        idxs = []
        for i, (t0, t1) in enumerate(offsets):
            if t0 == t1:  # padding offsets (0,0)
                continue
            # overlap
            if (t0 < s1) and (t1 > s0):
                idxs.append(i)

        if not idxs:
            continue

        tags[idxs[0]] = f"B-{stype}"
        for j in idxs[1:]:
            tags[j] = f"I-{stype}"

    return [L.SLOT_TO_ID.get(t, L.SLOT_TO_ID["O"]) for t in tags]

class JaneGPTv3Dataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=128):
        self.items = load_jsonl(jsonl_path)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        text = ex["text"]
        domain = ex["domain"]
        action = ex["action"]
        spans = ex.get("spans", [])

        enc = self.tok.encode(text)
        ids = enc.ids[: self.max_len]
        offsets = enc.offsets[: self.max_len]

        attn_mask = [1] * len(ids)
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [PAD_ID] * pad_len
            attn_mask += [0] * pad_len
            offsets += [(0, 0)] * pad_len

        slot_ids = spans_to_bio(offsets, spans)
        slot_ids = [(sid if attn_mask[i] == 1 else -100) for i, sid in enumerate(slot_ids)]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels_domain": torch.tensor(L.DOMAIN_TO_ID[domain], dtype=torch.long),
            "labels_action": torch.tensor(L.ACTION_TO_ID[action], dtype=torch.long),
            "labels_slots": torch.tensor(slot_ids, dtype=torch.long),
            "text": text,
        }
