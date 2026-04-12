## Fair Benchmark Summary (Apr 2026)

Only schema-aligned or schema-agnostic benchmarks are included here.

### 1) Runtime Reliability (Janus)

| Metric | Result |
|---|---:|
| Total turns | 82 |
| Local commands | 67 |
| Llama routes | 3 |
| Clarifications | 12 |
| Runtime errors | **0** |

### 2) Latency (CUDA, batch=1)

| Model | Metric | Value |
|---|---|---:|
| JaneGPT-v2 | Avg prediction latency | 31.60 ms |
| JaneGPT-v2 | Throughput | 32 preds/sec |
| Janus | Forward mean | 35.37 ms |
| Janus | Forward p95 | 36.71 ms |
| Janus | End-to-end predict mean | 25.31 ms |
| Janus | End-to-end predict p95 | 34.60 ms |

### 3) OOD Rejection (Fair Cross-Domain Safety Test)

#### BANKING77

| Model | OOD Precision | OOD Recall | OOD F1 | In-scope False Positive Rate |
|---|---:|---:|---:|---:|
| JaneGPT-v2 | 99.35% | 89.75% | 94.31% | 20.00% |
| Janus | 100.00% | 78.25% | 87.80% | 0.00% |

#### CLINC OOS

| Model | OOD Precision | OOD Recall | OOD F1 | In-scope False Positive Rate |
|---|---:|---:|---:|---:|
| JaneGPT-v2 | 99.14% | 81.00% | 89.16% | 20.00% |
| Janus | 100.00% | 65.60% | 79.23% | 0.00% |

### Excluded From Headline Results

- MASSIVE and SNIPS mapped-intent accuracy is excluded from headline claims.
- Reason: partial label mapping between external intents and JaneGPT command schema can understate true model quality.

### Source Files

- tools/results/baseline_summary.json
- tools/results/public_benchmarks.json
