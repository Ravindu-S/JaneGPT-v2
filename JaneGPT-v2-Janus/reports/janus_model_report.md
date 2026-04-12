## Model Details (auto-generated)

- **Checkpoint**: `C:\Users\RAVINDU\Documents\Projects\JaneGPT-Core\JaneGPT-v2-Janus\weights\janegpt_v2_janus.pt`
- **Checkpoint size**: `30.62 MB`
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

### Inference Benchmark (forward-only)

| Stat | ms |
|---|---:|
| mean | 15.685 |
| p50 | 15.306 |
| p95 | 17.890 |

### Inference Benchmark (end-to-end predict)

| Stat | ms |
|---|---:|
| mean | 10.818 |
| p50 | 13.493 |
| p95 | 14.899 |
