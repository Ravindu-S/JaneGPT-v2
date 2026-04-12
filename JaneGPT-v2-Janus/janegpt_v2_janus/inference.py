import os, re
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .multitask import JaneGPTv3MultiTask
from . import labels as L

PAD_ID = 0

# Follow-ups (volume/brightness)
FOLLOWUP_UP = re.compile(r"\b(not enough|more|increase it|raise it|turn it up|louder|boost it|a bit more|again more)\b", re.I)
FOLLOWUP_DOWN = re.compile(r"\b(too loud|too bright|less|decrease it|lower it|turn it down|quieter|softer|a bit less|dimmer)\b", re.I)
FOLLOWUP_REPEAT = re.compile(r"\b(again|do it again|repeat that|same again)\b", re.I)

# Smalltalk/gratitude safety net (pre-NLU)
SMALLTALK = re.compile(r"^\s*(hi|hello|hey|yo|sup|thanks|thank you|thx|ty)\b", re.I)

def _decode_bio(pred_ids, offsets, text):
    spans = []
    cur = None  # (stype, start, end, token_idxs)

    def flush():
        nonlocal cur
        if cur is not None:
            stype, s0, s1, toks = cur
            if s0 is not None and s1 is not None and s0 < s1:
                spans.append((stype, s0, s1, toks))
            cur = None

    for i, sid in enumerate(pred_ids):
        tag = L.ID_TO_SLOT[int(sid)]
        t0, t1 = offsets[i]
        if t0 == t1:
            flush()
            continue
        if tag == "O":
            flush()
            continue

        prefix, stype = tag.split("-", 1)
        if prefix == "B":
            flush()
            cur = (stype, t0, t1, [i])
        else:  # I
            if cur is None or cur[0] != stype:
                flush()
                cur = (stype, t0, t1, [i])
            else:
                cur = (cur[0], cur[1], t1, cur[3] + [i])

    flush()

    out = {}
    for stype, s0, s1, toks in spans:
        stext = text[s0:s1].strip()
        if not stext:
            continue
        if stype not in out or (s1 - s0) > (out[stype]["end"] - out[stype]["start"]):
            out[stype] = {"text": stext, "start": s0, "end": s1, "token_idxs": toks}
    return out

class JaneGPTv3NLU:
    def __init__(self, model_path, tokenizer_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = Tokenizer.from_file(tokenizer_path)

        ckpt = torch.load(model_path, map_location="cpu")
        cfg = ckpt.get("config", {})
        self.max_len = int(cfg.get("max_len", 96))

        self.model = JaneGPTv3MultiTask(
            num_domains=len(L.DOMAIN_LABELS),
            num_actions=len(L.ACTION_LABELS),
            num_slot_labels=len(L.SLOT_LABELS),
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()

    def _encode(self, text):
        enc = self.tok.encode(text)
        ids = enc.ids[: self.max_len]
        offsets = enc.offsets[: self.max_len]

        attn = [1] * len(ids)
        pad = self.max_len - len(ids)
        if pad > 0:
            ids += [PAD_ID] * pad
            attn += [0] * pad
            offsets += [(0, 0)] * pad

        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        m = torch.tensor([attn], dtype=torch.long, device=self.device)
        return x, m, offsets

    def _followup_override(self, text, state):
        if not state:
            return None
        last_domain = state.get("last_domain")
        last_action = state.get("last_action")

        if last_domain in ("volume", "brightness"):
            if FOLLOWUP_UP.search(text):
                return {"domain": last_domain, "action": "up", "reason": "followup_adjust_up"}
            if FOLLOWUP_DOWN.search(text):
                return {"domain": last_domain, "action": "down", "reason": "followup_adjust_down"}
            if FOLLOWUP_REPEAT.search(text) and last_action:
                return {"domain": last_domain, "action": last_action, "reason": "followup_repeat"}
        else:
            if FOLLOWUP_REPEAT.search(text) and last_domain and last_action:
                return {"domain": last_domain, "action": last_action, "reason": "followup_repeat"}

        return None

    def _best_allowed_pair(self, dom_probs, act_probs, k_dom=3, k_act=5):
        # returns (domain, action, score)
        dom_topv, dom_topi = torch.topk(dom_probs, k=min(k_dom, dom_probs.numel()))
        act_topv, act_topi = torch.topk(act_probs, k=min(k_act, act_probs.numel()))

        best = None
        for di, dp in zip(dom_topi.tolist(), dom_topv.tolist()):
            domain = L.ID_TO_DOMAIN[int(di)]
            for ai, ap in zip(act_topi.tolist(), act_topv.tolist()):
                action = L.ID_TO_ACTION[int(ai)]
                if hasattr(L, "is_allowed_pair") and not L.is_allowed_pair(domain, action):
                    continue
                score = float(dp * ap)
                if (best is None) or (score > best[2]):
                    best = (domain, action, score)

        return best  # or None

    @torch.no_grad()
    def predict(self, text, state=None, conf_thresh=0.65, margin_thresh=0.15, slot_conf_thresh=0.60):
        text = text.strip()

        # hard safety for smalltalk/thanks
        if SMALLTALK.match(text):
            return {
                "type": "command",
                "domain": "conversation",
                "action": "chat",
                "slots": {},
                "confidence": 1.0,
                "domain_conf": 1.0,
                "action_conf": 1.0,
                "legacy_intent": L.to_legacy_intent("conversation", "chat") if hasattr(L, "to_legacy_intent") else None,
                "route": "llama",
                "notes": {"smalltalk_override": True},
            }

        # follow-up override
        ov = self._followup_override(text, state)
        if ov is not None:
            return {
                "type": "command",
                "domain": ov["domain"],
                "action": ov["action"],
                "slots": {},
                "confidence": 1.0,
                "notes": {"followup": ov["reason"]},
            }

        x, m, offsets = self._encode(text)
        out = self.model(x, attention_mask=m, causal=False)

        dom_probs = F.softmax(out["logits_domain"][0], dim=-1)
        act_probs = F.softmax(out["logits_action"][0], dim=-1)

        # margins (top1-top2) for uncertainty gating
        dom_topv, dom_topi = torch.topk(dom_probs, k=2)
        act_topv, act_topi = torch.topk(act_probs, k=2)

        p_dom = float(dom_topv[0].item())
        p_act = float(act_topv[0].item())
        dom_margin = float((dom_topv[0] - dom_topv[1]).item()) if dom_topv.numel() > 1 else 1.0
        act_margin = float((act_topv[0] - act_topv[1]).item()) if act_topv.numel() > 1 else 1.0

        # choose best allowed pair among top-k (fixes "thanks" issues)
        best = self._best_allowed_pair(dom_probs, act_probs, k_dom=3, k_act=6)
        if best is None:
            return {"type": "clarify", "question": "I’m not sure what you want. Can you rephrase?",
                    "debug": {"reason": "no_allowed_pair"}}

        domain, action, pair_score = best

        # gating
        if (min(p_dom, p_act) < conf_thresh) or (min(dom_margin, act_margin) < margin_thresh):
            return {
                "type": "clarify",
                "question": "I’m not sure I understood. Can you rephrase?",
                "debug": {"domain": domain, "action": action, "p_dom": p_dom, "p_act": p_act, "dom_margin": dom_margin, "act_margin": act_margin},
            }

        # slots
        slot_logits = out["logits_slots"][0]  # (T,C)
        slot_probs = F.softmax(slot_logits, dim=-1)
        slot_pred = slot_probs.argmax(dim=-1).tolist()

        raw_slots = _decode_bio(slot_pred, offsets, text)

        slot_out = {}
        for stype, info in raw_slots.items():
            idxs = info["token_idxs"]
            probs = []
            for ti in idxs:
                pred_class = int(slot_pred[ti])
                probs.append(float(slot_probs[ti, pred_class].item()))
            conf = sum(probs) / max(len(probs), 1)
            slot_out[stype] = {"text": info["text"], "start": info["start"], "end": info["end"], "confidence": conf}

        # required slots clarification
        req = getattr(L, "REQUIRED_SLOTS", {}).get((domain, action), [])
        for r in req:
            if r == "TIME|DURATION":
                if ("TIME" not in slot_out) and ("DURATION" not in slot_out):
                    return {"type": "clarify", "question": "When should I set the reminder (time or duration)?",
                            "debug": {"domain": domain, "action": action, "reason": "missing_time_or_duration"}}
            else:
                if r not in slot_out:
                    q = {
                        "VALUE": "What value should I set it to?",
                        "APP_NAME": "Which app?",
                        "QUERY": "What should I search for?",
                        "WINDOW_NAME": "Which window should I focus?",
                        "TEXT": "What text should I use?",
                    }.get(r, f"Please provide {r}.")
                    return {"type": "clarify", "question": q, "debug": {"domain": domain, "action": action, "reason": f"missing_{r}"}}

                if slot_out[r]["confidence"] < slot_conf_thresh:
                    return {"type": "clarify",
                            "question": f"I’m not sure I got the {r.lower().replace('_',' ')}. Can you repeat it?",
                            "debug": {"domain": domain, "action": action, "reason": f"lowconf_{r}", "slot": slot_out[r]}}

        confidence = float(pair_score) ** 0.5
        legacy = L.to_legacy_intent(domain, action) if hasattr(L, "to_legacy_intent") else None

        return {
            "type": "command",
            "domain": domain,
            "action": action,
            "slots": slot_out,
            "confidence": confidence,
            "domain_conf": p_dom,
            "action_conf": p_act,
            "legacy_intent": legacy,
            "route": "llama" if (domain == "conversation" and action == "chat") else "local",
        }

    def update_state(self, result, state=None):
        state = dict(state or {})
        if result.get("type") == "command":
            state["last_domain"] = result.get("domain")
            state["last_action"] = result.get("action")
            state["last_slots"] = result.get("slots", {})
        return state
