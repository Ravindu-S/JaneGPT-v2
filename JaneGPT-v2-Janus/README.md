# JaneGPT-v2-Janus - Hierarchical NLU with State-Aware Runtime

A lightweight [Transformer](../assets/TECHNICAL_DICTIONARY.md#transformer) [NLU](../assets/TECHNICAL_DICTIONARY.md#nlu-natural-language-understanding) stack for assistant commands with robust follow-ups, [clarification loops](../assets/TECHNICAL_DICTIONARY.md#slot-clarification--clarification-loops), and local-first execution.

**7.95M parameters | 10 [domains](../assets/TECHNICAL_DICTIONARY.md#domain) + 33 [actions](../assets/TECHNICAL_DICTIONARY.md#action) + 15 [BIO slot](../assets/TECHNICAL_DICTIONARY.md#bio-tagging-begin-inside-outside) labels | ~30.6 MB checkpoint | production-style runtime included**

![JaneGPT AI Banner](../assets/banner.webp)

JaneGPT-v2-Janus predicts structured actions as `(domain, action, slots)` and adds runtime intelligence for real multi-turn use.

Janus behavior highlights:
- "set volume" -> asks for missing value
- "55" -> resolves pending [slot](../assets/TECHNICAL_DICTIONARY.md#slot-slot-filling) and executes `volume/set`
- "that is not enough" -> uses prior state to adjust up
- "hello" -> routes to chat model, not local command executor

---

## Why This Model Is Strong

- Built from scratch (not a fine-tuned external assistant model).
- Hierarchical command understanding: [domain](../assets/TECHNICAL_DICTIONARY.md#domain) + [action](../assets/TECHNICAL_DICTIONARY.md#action) + [slots](../assets/TECHNICAL_DICTIONARY.md#slot-slot-filling).
- Runtime-safe [clarification logic](../assets/TECHNICAL_DICTIONARY.md#slot-clarification--clarification-loops) for missing required inputs.
- [Stateful follow-ups](../assets/TECHNICAL_DICTIONARY.md#stateful-follow-ups) for natural multi-turn corrections.
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
- `type="command"` with `domain`, `action`, `slots`, and [confidence](../assets/TECHNICAL_DICTIONARY.md#confidence-gating)
- `type="clarify"` with a targeted [clarification](../assets/TECHNICAL_DICTIONARY.md#slot-clarification--clarification-loops) follow-up question

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

- [Domains](../assets/TECHNICAL_DICTIONARY.md#domain) (10): `volume`, `brightness`, `media`, `apps`, `browser`, `productivity`, `screen`, `window`, `system`, `conversation`
- [Actions](../assets/TECHNICAL_DICTIONARY.md#action) (33):
  - Adjustments: `up`, `down`, `set`, `mute`, `unmute`
  - Media: `play`, `pause`, `next`, `previous`
  - Apps: `launch`, `close`, `switch`
  - Other: `search`, `set_reminder`, `screenshot`, `read`, `explain`, `undo`, `quit`, `chat`, `minimize`, `maximize`, `restore`, `focus`, `copy`, `paste`, `cut`, `lock`, `sleep`, `wifi_on`, `wifi_off`, `bluetooth_on`, `bluetooth_off`
- [Slot types](../assets/TECHNICAL_DICTIONARY.md#slot-slot-filling) ([BIO](../assets/TECHNICAL_DICTIONARY.md#bio-tagging-begin-inside-outside), 15 labels): `VALUE`, `APP_NAME`, `QUERY`, `DURATION`, `TIME`, `WINDOW_NAME`, `TEXT`

Required-slot [clarification](../assets/TECHNICAL_DICTIONARY.md#slot-clarification--clarification-loops) rules are enforced (for example: `volume/set -> VALUE`, `apps/launch -> APP_NAME`, `browser/search -> QUERY`, `set_reminder -> TIME|DURATION`, `window/focus -> WINDOW_NAME`).

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
- 8 attention heads, 4 KV heads ([GQA](../assets/TECHNICAL_DICTIONARY.md#gqa-grouped-query-attention))
- [RoPE](../assets/TECHNICAL_DICTIONARY.md#rope-rotary-position-embedding) + [RMSNorm](../assets/TECHNICAL_DICTIONARY.md#rmsnorm-root-mean-square-layer-normalization) + [SwiGLU](../assets/TECHNICAL_DICTIONARY.md#swiglu-swish-gated-linear-unit)
- [Bidirectional attention](../assets/TECHNICAL_DICTIONARY.md#bidirectional-attention) mode for [NLU](../assets/TECHNICAL_DICTIONARY.md#nlu-natural-language-understanding)

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
| `runtime/jane_nlu_runtime.py` | Assistant runtime wrapper. Manages dialogue state, [clarification loops](../assets/TECHNICAL_DICTIONARY.md#slot-clarification--clarification-loops), pending [slot](../assets/TECHNICAL_DICTIONARY.md#slot-slot-filling) resolution, safe slot cleanup, colloquial overrides, and chat/local routing. |
| `janegpt_v2_janus/inference.py` | Core inference API (`JaneGPTv3NLU`). Encodes text, loads `.pt` and tokenizer, predicts [domain](../assets/TECHNICAL_DICTIONARY.md#domain)/[action](../assets/TECHNICAL_DICTIONARY.md#action)/[slots](../assets/TECHNICAL_DICTIONARY.md#slot-slot-filling), applies [confidence gating](../assets/TECHNICAL_DICTIONARY.md#confidence-gating) and follow-up overrides. |
| `janegpt_v2_janus/multitask.py` | Multi-head model wrapper. Produces domain [logits](../assets/TECHNICAL_DICTIONARY.md#logits), action [logits](../assets/TECHNICAL_DICTIONARY.md#logits), and token-level [slot](../assets/TECHNICAL_DICTIONARY.md#slot-slot-filling) [logits](../assets/TECHNICAL_DICTIONARY.md#logits) from shared backbone features. |
| `janegpt_v2_janus/architecture.py` | [Transformer](../assets/TECHNICAL_DICTIONARY.md#transformer) backbone implementation ([RMSNorm](../assets/TECHNICAL_DICTIONARY.md#rmsnorm-root-mean-square-layer-normalization), [RoPE](../assets/TECHNICAL_DICTIONARY.md#rope-rotary-position-embedding), [GQA](../assets/TECHNICAL_DICTIONARY.md#gqa-grouped-query-attention), [SwiGLU](../assets/TECHNICAL_DICTIONARY.md#swiglu-swish-gated-linear-unit), blocks, attention mask behavior). |
| `janegpt_v2_janus/labels.py` | Schema and policy source of truth: labels, allowed [domain](../assets/TECHNICAL_DICTIONARY.md#domain)-[action](../assets/TECHNICAL_DICTIONARY.md#action) combinations, required [slot](../assets/TECHNICAL_DICTIONARY.md#slot-slot-filling) rules, and optional legacy intent mapping. |
| `janegpt_v2_janus/dataset.py` | Training/eval dataset utilities for JSONL, [BIO](../assets/TECHNICAL_DICTIONARY.md#bio-tagging-begin-inside-outside) conversion, attention masking, and supervised labels. |

---

## Performance Snapshot

<p align="center">
   <img src="https://img.shields.io/badge/Runtime_Suite-82_turns%20%7C%200_errors-16a34a?style=for-the-badge" alt="Runtime Suite 82 Turns 0 Errors" />
   <img src="https://img.shields.io/badge/Predict_Latency-25.31ms_mean-0ea5e9?style=for-the-badge" alt="Predict Latency 25.31ms" />
   <img src="https://img.shields.io/badge/OOD_F1-BANKING77_87.80%25-1f6feb?style=for-the-badge" alt="OOD F1 Banking77 87.80 percent" />
</p>

- Parameters: 7,949,626
- Backbone params: 7,803,136
- Head params: 146,490
- Checkpoint: `weights/janegpt_v2_janus.pt` (~30.62 MB)
- Tokenizer: custom [BPE](../assets/TECHNICAL_DICTIONARY.md#bpe-byte-pair-encoding-tokenizer) (`weights/tokenizer.json`, vocab 8192)

**Understanding These Benchmarks:**

| Benchmark | What It Tests | What It Means | Example |
|-----------|--------------|---------------|----------|
| **Runtime Reliability** | Can Janus handle 82 multi-turn conversations without crashing? | 0 errors = production-ready; 10+ errors = unstable. Tests real assistant behavior ([clarifications](../assets/TECHNICAL_DICTIONARY.md#slot-clarification--clarification-loops), [slot filling](../assets/TECHNICAL_DICTIONARY.md#slot-slot-filling), state changes) | Turn 1: "Set volume" → Turn 45: "Actually make it louder" → Turn 82: Still perfect |
| **[Latency](../assets/TECHNICAL_DICTIONARY.md#latency)** | How fast Janus runs per prediction | Speed is critical for real-time assistants. Under 50ms = excellent; over 200ms = noticeable lag | User says "open chrome" → model responds in ~25ms |
| **[OOD](../assets/TECHNICAL_DICTIONARY.md#out-of-domain-ood) Safety (BANKING77)** | Can Janus reject finance questions when trained on home automation? | Tests this model's judgment. ~90% [F1](../assets/TECHNICAL_DICTIONARY.md#f1-score) = excellent (rejects what it shouldn't handle). Under 60% = dangerous (would give wrong answers) | User asks "What's my account balance?" → Janus correctly says "I can't help with that" |
| **[OOD](../assets/TECHNICAL_DICTIONARY.md#out-of-domain-ood) Safety (CLINC)** | Can Janus reject random real-world off-topic requests? | Similar to BANKING77 but with diverse random questions. Proves this model knows its limits | User asks "What's the capital of France?" → Janus correctly rejects it |

### Fair Runtime Reliability (in-domain suite)

| Metric | Result |
|---|---:|
| Total turns | 82 |
| Local commands | 67 |
| Llama routes | 3 |
| Clarifications | 12 |
| Runtime errors | **0** |

### Fair Latency (CUDA, batch=1, environment-dependent)

| Metric | Value |
|---|---:|
| Forward mean | 35.37 ms |
| Forward p95 | 36.71 ms |
| End-to-end predict mean | 25.31 ms |
| End-to-end predict p95 | 34.60 ms |

### Fair OOD Rejection (schema-agnostic safety)

| Dataset | OOD [Precision](../assets/TECHNICAL_DICTIONARY.md#precision) | OOD [Recall](../assets/TECHNICAL_DICTIONARY.md#recall) | OOD [F1](../assets/TECHNICAL_DICTIONARY.md#f1-score) | In-scope False Positive Rate |
|---|---:|---:|---:|---:|
| BANKING77 | 100.00% | 78.25% | 87.80% | 0.00% |
| CLINC OOS | 100.00% | 65.60% | 79.23% | 0.00% |

⚠️ **Why we exclude MASSIVE/SNIPS from headlines:**  
Janus was trained on assistant commands like "set_volume" and "open_app".  
MASSIVE/SNIPS use different command names ("light_on", "alarm_set", etc.).  
Only ~50% of their labels could be mapped, so accuracy scores would mislead about Janus's real quality.  
Instead, the [OOD](../assets/TECHNICAL_DICTIONARY.md#out-of-domain-ood) safety tests (schema-independent) prove this model's actual judgment capability.

**Bottom Line:** Janus is **SOLID** ✅
- Fast enough for real users (25-35ms per prediction)
- Stable enough for production (0 crashes in 82 turns)
- Safe enough to deploy (87-88% [OOD](../assets/TECHNICAL_DICTIONARY.md#out-of-domain-ood) rejection accuracy)

For full reports, see:
- `reports/janus_model_report.md`
- `reports/janus_model_report.json`
- `reports/public_benchmarks.json`
- `reports/fair_benchmarks.md`

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

Apache License 2.0 (see [LICENSE](LICENSE)).

## Author

Ravindu Senanayake
