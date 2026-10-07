## Model Details (auto-generated)

- **Checkpoint**: `weights/janegpt_v2_janus.pt`
- **Checkpoint size**: `30.61 MB`
- **Device used for benchmark**: `cuda`

### Architecture

| Property | Value |
|---|---:|
| vocab_size | 8192 |
| embed_dim | 256 |
| num_layers | 8 |
| num_heads | 8 |
| num_kv_heads | 4 |
| ff_hidden | 672 |
| max_len | 96 |
| causal | False |

### Parameters

| Component | Params |
|---|---:|
| Total | 7,949,626 |
| Trainable | 7,949,626 |
| Backbone | 7,803,136 |
| Heads (domain+action+slots) | 146,490 |

### Label Set Sizes

| Label set | Count |
|---|---:|
| Domains | 10 |
| Actions | 33 |
| Slot labels (BIO) | 15 |

### Validation Metrics

| Metric | Value |
|---|---:|
| val_loss | 0.130094 |
| domain_acc | 0.988668 |
| action_acc | 0.984686 |
| pair_acc | 0.984074 |
| slot_precision | 0.988701 |
| slot_recall | 0.998004 |
| slot_f1 | 0.993330 |
| val_examples | 3265 |

### Inference Benchmark (forward-only)

| Stat | ms |
|---|---:|
| mean | 22.232 |
| p50 | 21.256 |
| p95 | 28.662 |

### Inference Benchmark (end-to-end predict)

| Stat | ms |
|---|---:|
| mean | 15.372 |
| p50 | 18.221 |
| p95 | 23.734 |
