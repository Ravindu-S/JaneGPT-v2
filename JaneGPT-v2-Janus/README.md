# JaneGPT-v2-Janus - Hierarchical NLU with State-Aware Runtime

A lightweight Transformer NLU stack for assistant commands with robust follow-ups, clarification loops, and local-first execution.

**7.95M parameters | 10 domains + 33 actions + 15 BIO slot labels | ~30.6 MB checkpoint | production-style runtime included**

![JaneGPT AI Banner](../assets/banner.webp)

JaneGPT-v2-Janus predicts structured actions as `(domain, action, slots)` and adds runtime intelligence for real multi-turn use.

Janus behavior highlights:
- "set volume" -> asks for missing value
- "55" -> resolves pending slot and executes `volume/set`
- "that is not enough" -> uses prior state to adjust up
- "hello" -> routes to chat model, not local command executor

---

## Why This Model Is Strong

- Built from scratch (not a fine-tuned external assistant model).
- Hierarchical command understanding: domain + action + slots.
- Runtime-safe clarification logic for missing required inputs.
- Stateful follow-ups for natural multi-turn corrections.
- Hybrid routing: local control vs. chat-model deferral when appropriate.
- Fast and compact for practical desktop/edge assistant use.

Latest internal runtime suite (`examples/demo_runtime_suite.py`):
- 82 turns
- 67 local command resolutions
- 3 chat routes (expected conversational turns)
- 12 clarifications
- 0 runtime errors

---

## Core Capabilities

### Prediction Output

The model returns one of:
- `type="command"` with `domain`, `action`, `slots`, and confidence
- `type="clarify"` with a targeted follow-up question

Command example:
```json
{
  "type": "command",
  "domain": "apps",
  "action": "launch",
  "slots": {
    "APP_NAME": {"text": "chrome", "start": 5, "end": 11, "confidence": 0.999}
  },
  "route": "local"
}
```

Clarify example:
```json
{
  "type": "clarify",
  "question": "What value should I set it to?",
  "debug": {"domain": "volume", "action": "set", "reason": "missing_VALUE"}
}
```

### Labels and Schema

- Domains (10): `volume`, `brightness`, `media`, `apps`, `browser`, `productivity`, `screen`, `window`, `system`, `conversation`
- Actions (33):
  - Adjustments: `up`, `down`, `set`, `mute`, `unmute`
  - Media: `play`, `pause`, `next`, `previous`
  - Apps: `launch`, `close`, `switch`
  - Other: `search`, `set_reminder`, `screenshot`, `read`, `explain`, `undo`, `quit`, `chat`, `minimize`, `maximize`, `restore`, `focus`, `copy`, `paste`, `cut`, `lock`, `sleep`, `wifi_on`, `wifi_off`, `bluetooth_on`, `bluetooth_off`
- Slot types (BIO, 15 labels): `VALUE`, `APP_NAME`, `QUERY`, `DURATION`, `TIME`, `WINDOW_NAME`, `TEXT`

Required-slot clarification rules are enforced (for example: `volume/set -> VALUE`, `apps/launch -> APP_NAME`, `browser/search -> QUERY`, `set_reminder -> TIME|DURATION`, `window/focus -> WINDOW_NAME`).

---

## Architecture Diagram

```text
Input Text
    │
    ▼
┌──────────────────────┐
│   BPE Tokenizer      │  vocab_size=8192, pad to 96 tokens
│  (CustomTokenizer)   │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Token Embedding     │  [seq_len, embed_dim=256]
│   + RoPE Position    │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  8x Transformer      │  Each block:
│  Bidirectional       │  ├─ RMSNorm
│  (no causal mask)    │  ├─ GQA (8 heads, 4 KV heads)
│                      │  ├─ RoPE per-layer
│                      │  ├─ RMSNorm
│                      │  └─ SwiGLU FFN (256->672->256)
└──────────┬───────────┘
           ▼
┌──────────────────────────────────────┐
│       RMSNorm (final layer)          │
└──────────┬───────────────────────────┘
           │
      ┌────┴────┬────────┬────────┐
      ▼         ▼        ▼        ▼
  Domain    Action    Slots   Confidence
   Head      Head      Head      Score
  (10)       (33)      (BIO)
                       (15)
      │         │        │        │
      └─────────┴────────┴────────┘
                 ▼
      Output: (domain, action, slots, confidence)
```

Backbone spec:
- 8 layers, embed dim 256
- 8 attention heads, 4 KV heads (GQA)
- RoPE + RMSNorm + SwiGLU
- Bidirectional attention mode for NLU

---

## Runtime + Model File Flow (Beginner Guide)

This is how Python files and the `.pt` checkpoint work together.

```text
User text
   │
   ▼
runtime/jane_nlu_runtime.py
   │  (conversation state, pending slot fill, safe overrides, routing)
   ▼
janegpt_v2_janus/inference.py (JaneGPTv3NLU)
   │
   ├─ loads tokenizer from weights/tokenizer.json
   ├─ loads checkpoint from weights/janegpt_v2_janus.pt
   └─ runs model forward + decoding
          │
          ▼
janegpt_v2_janus/multitask.py (JaneGPTv3MultiTask)
   │  (domain head + action head + slot head)
   ▼
janegpt_v2_janus/architecture.py (JaneGPTBackbone)
   │  (Transformer blocks: RoPE/GQA/SwiGLU/RMSNorm)
   ▼
janegpt_v2_janus/labels.py
   (domain/action vocab, slot labels, allowed pairs, required slots, legacy mapping)
```

### What Each File Does

| File | Responsibility |
|---|---|
| `runtime/jane_nlu_runtime.py` | Assistant runtime wrapper. Manages dialogue state, clarification loops, pending slot resolution, safe slot cleanup, colloquial overrides, and chat/local routing. |
| `janegpt_v2_janus/inference.py` | Core inference API (`JaneGPTv3NLU`). Encodes text, loads `.pt` and tokenizer, predicts domain/action/slots, applies confidence gating and follow-up overrides. |
| `janegpt_v2_janus/multitask.py` | Multi-head model wrapper. Produces domain logits, action logits, and token-level slot logits from shared backbone features. |
| `janegpt_v2_janus/architecture.py` | Transformer backbone implementation (RMSNorm, RoPE, GQA, SwiGLU, blocks, attention mask behavior). |
| `janegpt_v2_janus/labels.py` | Schema and policy source of truth: labels, allowed domain-action combinations, required slot rules, and optional legacy intent mapping. |
| `janegpt_v2_janus/dataset.py` | Training/eval dataset utilities for JSONL, BIO conversion, attention masking, and supervised labels. |

---

## Performance Snapshot

- Parameters: 7,949,626
- Backbone params: 7,803,136
- Head params: 146,490
- Checkpoint: `weights/janegpt_v2_janus.pt` (~30.62 MB)
- Tokenizer: custom BPE (`weights/tokenizer.json`, vocab 8192)

Benchmark summaries (environment-dependent):
- CPU forward mean: ~15.58 ms
- CPU end-to-end predict mean: ~9.36 ms
- CUDA forward mean: ~15.69 ms
- CUDA end-to-end predict mean: ~10.82 ms

For full auto-generated benchmark and parameter report, see:
- `reports/janus_model_report.md`
- `reports/janus_model_report.json`

---

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1) Raw NLU demo

```bash
python examples/demo_inference.py
```

Shows direct model outputs (`domain/action/slots`) without runtime state.

### 2) Assistant runtime demo (recommended)

```bash
python examples/demo_runtime.py
```

Shows full runtime behavior: clarifications, pending slot filling, follow-up state, and routing.

---

## Python Usage

### Basic NLU

```python
from janegpt_v2_janus.inference import JaneGPTv3NLU

nlu = JaneGPTv3NLU(
    model_path="weights/janegpt_v2_janus.pt",
    tokenizer_path="weights/tokenizer.json",
)

state = {}
result = nlu.predict("open chrome", state=state)
print(result)
```

### Assistant Runtime

```python
from runtime.jane_nlu_runtime import JaneNLURuntime

rt = JaneNLURuntime(base_dir=".")
state = {}

out, state = rt.handle_turn("set volume", state)
print(out["say"])  # What value should I set it to?

out, state = rt.handle_turn("55", state)
print(out["do"])   # local volume/set with VALUE=55
```

---

## Repository Essentials

Minimum inference files:
- `weights/janegpt_v2_janus.pt`
- `weights/tokenizer.json`
- `janegpt_v2_janus/architecture.py`
- `janegpt_v2_janus/multitask.py`
- `janegpt_v2_janus/labels.py`
- `janegpt_v2_janus/inference.py`

Runtime wrapper:
- `runtime/jane_nlu_runtime.py`

Reports and diagnostics:
- `reports/*.md`, `reports/*.json`, `reports/*.png`

---

## Limitations

- This is an NLU command model, not a generative chat model.
- Non-command conversation should be routed to an LLM/chat backend.
- Language coverage follows training data distribution (primarily English).
- Slot semantics depend on data annotation policy.

---

## License

Apache License 2.0 (see `LICENSE`).

## Author

Ravindu Senanayake
