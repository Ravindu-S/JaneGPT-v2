# Model Card: JaneGPT v2 Intent Classifier

## Model Overview

| Field | Value |
|-------|-------|
| Model Name | JaneGPT v2 Intent Classifier |
| Version | 2.0 |
| Type | Decoder-only Transformer + Classification Head |
| Task | Intent Classification (22 classes) |
| Language | English |
| Parameters | ~7.8M |
| Created By | Ravindu Senanayake |
| License | MIT |

## Intended Use

- Classifying user commands for virtual assistants
- Routing voice/text commands to appropriate handlers
- Edge deployment where large LLMs are impractical

## Out of Scope

- Text generation / conversation
- Languages other than English
- Commands outside the 22 supported categories
- Long document classification

## Architecture

Decoder-only transformer with:
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA) — 8 heads, 4 KV heads
- SwiGLU feed-forward networks
- RMSNorm normalization
- BPE tokenizer (8192 vocab)
- Classification head: Linear → GELU → Dropout → Linear

## Training

- **Pre-training**: Language modeling on general text
- **Fine-tuning**: Intent classification on ~2,400 labeled examples
- **Backbone freeze**: First 2 epochs (head only), then full fine-tuning
- **Optimizer**: AdamW (lr=2e-4, weight_decay=0.01)
- **Schedule**: Cosine decay with warmup
- **Epochs**: 10
- **Best checkpoint**: Selected by validation accuracy

## Performance

- Training accuracy: ~96.7%
- Validation accuracy: 88.6%
- Validation loss: (see checkpoint)

## Ethical Considerations

- Model makes classification decisions that could trigger system actions
- Should be paired with confirmation mechanisms for destructive actions
- No personal data is stored in model weights
- Training data included diverse English phrasing patterns

## Limitations

- Overfits somewhat (96.7% train vs 88.6% val)
- Weak on highly ambiguous short phrases without context
- Context-dependent phrases ("not enough", "the other one") need conversation history
- Not suitable for open-ended classification beyond 22 categories
