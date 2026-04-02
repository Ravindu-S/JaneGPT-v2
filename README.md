# JaneGPT v2 — Intent Classification Model

A lightweight, fast, and accurate intent classification model built from scratch for virtual assistant command understanding.

**7.8M parameters | 22 intent classes | 88.6% accuracy | ~50ms inference on GPU**


![JANE AI Banner](assets/banner.webp)


## Overview

JaneGPT v2 is a decoder-only transformer with a classification head, designed to understand user commands for a virtual assistant. It classifies natural language into 22 intent categories covering volume control, brightness, media playback, app management, browser search, and more.

Built with modern architecture techniques used in state-of-the-art LLMs, scaled down to run efficiently on consumer hardware.

## Key Features

- **From scratch** — Not fine-tuned from another model. Architecture and training done from zero.
- **Tiny but capable** — 7.8M parameters (1000x smaller than Llama 3 8B)
- **Fast** — ~50ms on GPU, ~200ms on CPU
- **Modern architecture** — RoPE, Grouped Query Attention, SwiGLU, RMSNorm
- **Context-aware** — Accepts conversation context for multi-turn understanding
- **Practical** — Designed for real-world virtual assistant deployment

## Supported Intents (22 classes)

| Category | Intents |
|----------|---------|
| Volume | `volume_up`, `volume_down`, `volume_set`, `volume_mute` |
| Brightness | `brightness_up`, `brightness_down`, `brightness_set` |
| Media | `media_play`, `media_pause`, `media_next`, `media_previous` |
| Apps | `app_launch`, `app_close`, `app_switch` |
| Browser | `browser_search` |
| Productivity | `set_reminder`, `screenshot` |
| Screen | `read_screen`, `explain_screen` |
| Control | `undo`, `quit_jane` |
| Conversation | `chat` |

## Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/JaneGPT-v2-Intent-Classifier.git
cd JaneGPT-v2-Intent-Classifier
pip install -r requirements.txt
```

### Basic Usage

```python
from model.classifier import JaneGPTClassifier

classifier = JaneGPTClassifier()

intent, confidence = classifier.predict("turn up the volume")
print(f"Intent: {intent}, Confidence: {confidence:.2%}")
# Output: Intent: volume_up, Confidence: 86.10%

intent, confidence = classifier.predict("open chrome")
print(f"Intent: {intent}, Confidence: {confidence:.2%}")
# Output: Intent: app_launch, Confidence: 98.10%
```

### With Context

```python

# After a volume action, user says "not enough"
intent, confidence = classifier.predict(
    "not enough",
    context={"last_intent": "volume_up"}
)
# Output: Intent: volume_up, Confidence: 79.00%
```

### Top-K Predictions

```python

results = classifier.predict_top_k("play something", k=3)
for intent, conf in results:
    print(f"  {intent}: {conf:.2%}")
```

### Architecture

```text

Input Text
    │
    ▼
┌──────────────────┐
│  BPE Tokenizer   │  vocab_size=8192, pad to 128 tokens
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Token Embedding  │  [8192, 256]
└────────┬─────────┘
         ▼
┌──────────────────┐
│  8x Transformer  │  Each block:
│     Blocks       │  ├─ RMSNorm
│                  │  ├─ GQA (8 heads, 4 KV heads)
│                  │  ├─ RoPE (per-layer)
│                  │  ├─ RMSNorm
│                  │  └─ SwiGLU FFN (256 → 672 → 256)
└────────┬─────────┘
         ▼
┌──────────────────┐
│    RMSNorm       │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Last Token      │  Pool: use final token hidden state
│  Pooling         │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Classification   │  Linear(256,256) → GELU → Dropout → Linear(256,22)
│     Head         │
└────────┬─────────┘
         ▼
    22-class softmax
```

### Model Details

| Property | Value |
|----------|-------|
| Parameters | ~7.8M |
| Embedding dim | 256 |
| Attention heads | 8 |
| KV heads (GQA) | 4 |
| Layers | 8 |
| FF hidden dim | 672 |
| Max sequence length | 256 |
| Vocab size | 8,192 |
| Tokenizer | BPE |
| Training accuracy | ~96.7% |
| Validation accuracy | 88.6% |
| Checkpoint size | ~30 MB |

### Performance

| Input | Predicted Intent | Confidence |
|-------|------------------|------------|
| "increase the volume" | volume_up | 86% |
| "make it louder" | volume_up | 90% |
| "turn down the brightness" | brightness_down | 80% |
| "open chrome" | app_launch | 98% |
| "play some music" | media_play | 96% |
| "search for cats on youtube" | browser_search | 94% |
| "set a reminder for 5 minutes" | set_reminder | 96% |
| "take a screenshot" | screenshot | 88% |
| "shut down" | quit_jane | 85% |
| "hello" | chat | 97% |
| "undo that" | undo | 92% |
| "read this for me" | read_screen | 85% |
| "explain what's on my screen" | explain_screen | 87% |

### Input Format

The model expects input formatted as:

```text
user: <user text>
context: <context string or "none">
jane:
```

The classifier wrapper handles this formatting automatically.

### Requirements

- Python 3.10+
- PyTorch 2.0+
- tokenizers (HuggingFace)

### Limitations

- Not a conversational model — classifies intent only, does not generate text
- 22 classes only — commands outside the supported set will be classified as chat
- English only — trained primarily on English commands
- Short inputs — optimized for short commands (1-15 words), not paragraphs
- No entity extraction — returns intent label only, not structured entities

### Use Cases

- Virtual assistant command routing
- Smart home intent classification
- Voice command understanding
- Chatbot intent detection
- Edge device deployment (small enough for embedded systems)

### License

MIT License — free for personal and commercial use.

### Created By

Ravindu Senanayake

Built from scratch — architecture, training data, and training pipeline.