<a id="top"></a>

<p align="center">
  <img src="assets/janegpt-core-glitch.webp" alt="JaneGPT Core" width="760" />
</p>

<p align="center"><strong>Assistant-focused model suite for command understanding, hierarchical <a href="assets/TECHNICAL_DICTIONARY.md#nlu-natural-language-understanding">NLU</a>, and runtime-safe execution</strong></p>

<p align="center">
   <img src="https://img.shields.io/badge/models-2-1f6feb?style=for-the-badge" alt="Models 2" />
   <img src="https://img.shields.io/badge/featured-JaneGPT--v2--Janus-0ea5e9?style=for-the-badge" alt="Featured JaneGPT-v2-Janus" />
   <img src="https://img.shields.io/badge/updated-Apr_2026-16a34a?style=for-the-badge" alt="Updated Apr 2026" />
</p>

<p align="center">
   <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&pause=1300&color=0EA5E9&center=true&vCenter=true&width=820&lines=Lightweight+models+for+real+assistant+workflows;Local-first+NLU+with+clarification+and+follow-ups;Fast+to+run%2C+easy+to+extend" alt="Animated tagline" />
</p>

<p align="center">
   <a href="#start-here">Start Here</a> |
   <a href="#model-explorer">Model Explorer</a> |
   <a href="#quick-paths">Quick Paths</a> |
   <a href="#fair-benchmarks-apr-2026">Fair Benchmarks</a> |
   <a href="#current-model-list">Current Model List</a> |
   <a href="#add-a-new-model-in-60-seconds">Add New Model</a>
</p>

---

<p align="center">
  <img src="https://img.shields.io/badge/Model%20Lineup-Choose%20AI%20Your%20Engine-0f172a?style=for-the-badge&labelColor=111827&color=1f2937" alt="Model Lineup" />
</p>

<p align="center"><a href="JaneGPT-v2-Janus/README.md"><img src="assets/button-janus.svg" alt="JaneGPT-v2-Janus" /></a>&nbsp;&nbsp;<a href="JaneGPT-v2/README.md"><img src="assets/button-v2.svg" alt="JaneGPT-v2" /></a></p>

<p align="center"><img src="assets/caption-janus.svg" alt="Janus: Domain + Action + Slots + Stateful Follow-ups" />&nbsp;&nbsp;<img src="assets/caption-v2.svg" alt="v2: Fast 22-intent routing model" /></p>

---

## Start Here

For first-time visitors, begin with Janus:

- Featured model: [JaneGPT-v2-Janus](JaneGPT-v2-Janus/)
- Full docs: [JaneGPT-v2-Janus/README.md](JaneGPT-v2-Janus/README.md)
- Try now: [JaneGPT-v2-Janus/examples/demo_runtime.py](JaneGPT-v2-Janus/examples/demo_runtime.py)

If you want a simpler intent-only baseline:

- [JaneGPT-v2](JaneGPT-v2/)
- [JaneGPT-v2/README.md](JaneGPT-v2/README.md)

---

## Model Explorer

| Track | Best For | What You Get | Jump |
|---|---|---|---|
| **JaneGPT-v2-Janus** | Real assistant runtime flows | Hierarchical `(domain, action, slots)`, [clarifications](assets/TECHNICAL_DICTIONARY.md#slot-clarification--clarification-loops), pending-slot fill, follow-up state | [Open Janus](JaneGPT-v2-Janus/) |
| **JaneGPT-v2** | Fast intent routing baseline | Lightweight 22-intent classifier with simple integration | [Open v2](JaneGPT-v2/) |

<details>
<summary><strong>Expand: Janus quick navigation</strong></summary>

- Docs: [JaneGPT-v2-Janus/README.md](JaneGPT-v2-Janus/README.md)
- Runtime demo: [JaneGPT-v2-Janus/examples/demo_runtime.py](JaneGPT-v2-Janus/examples/demo_runtime.py)
- Inference demo: [JaneGPT-v2-Janus/examples/demo_inference.py](JaneGPT-v2-Janus/examples/demo_inference.py)
- Runtime source: [JaneGPT-v2-Janus/runtime/jane_nlu_runtime.py](JaneGPT-v2-Janus/runtime/jane_nlu_runtime.py)

</details>

<details>
<summary><strong>Expand: v2 quick navigation</strong></summary>

- Docs: [JaneGPT-v2/README.md](JaneGPT-v2/README.md)
- Basic inference: [JaneGPT-v2/examples/basic_inference.py](JaneGPT-v2/examples/basic_inference.py)
- Classifier wrapper: [JaneGPT-v2/model/classifier.py](JaneGPT-v2/model/classifier.py)

</details>

---

## Quick Paths

| I Want To... | Go Here |
|---|---|
| Understand Janus architecture and runtime behavior | [JaneGPT-v2-Janus/README.md](JaneGPT-v2-Janus/README.md) |
| Run assistant-style multi-turn behavior | [JaneGPT-v2-Janus/examples/demo_runtime.py](JaneGPT-v2-Janus/examples/demo_runtime.py) |
| Run pure model inference only | [JaneGPT-v2-Janus/examples/demo_inference.py](JaneGPT-v2-Janus/examples/demo_inference.py) |
| Use a simpler classifier baseline | [JaneGPT-v2/README.md](JaneGPT-v2/README.md) |
| Benchmark classifier performance | [JaneGPT-v2/examples/benchmark.py](JaneGPT-v2/examples/benchmark.py) |
| View fair benchmark summary | [JaneGPT-v2-Janus/reports/fair_benchmarks.md](JaneGPT-v2-Janus/reports/fair_benchmarks.md) |

---

## Fair Benchmarks (Apr 2026)

<p align="center">
   <img src="https://img.shields.io/badge/Janus_Runtime-82_turns%20%7C%200_errors-16a34a?style=for-the-badge" alt="Janus Runtime 82 Turns 0 Errors" />
   <img src="https://img.shields.io/badge/Janus_Predict-25.31ms_mean-0ea5e9?style=for-the-badge" alt="Janus Predict Mean 25.31ms" />
   <img src="https://img.shields.io/badge/v2_Predict-31.60ms_mean-1f6feb?style=for-the-badge" alt="v2 Predict Mean 31.60ms" />
</p>

Only schema-aligned or schema-agnostic benchmarks are shown here.

| Fair Test | JaneGPT-v2 | JaneGPT-v2-Janus | Why It Is Fair |
|---|---:|---:|---|
| [Latency](assets/TECHNICAL_DICTIONARY.md#latency) (CUDA, batch=1) | 31.60 ms mean, 32 preds/sec | 25.31 ms mean predict, 34.60 ms p95 | Same local hardware and same benchmark pipeline |
| Runtime reliability suite (82 turns) | - | 67 local commands, 3 Llama routes, 12 [clarifications](assets/TECHNICAL_DICTIONARY.md#slot-clarification--clarification-loops), **0 errors** | In-domain assistant behavior with strict pass/fail |
| [OOD](assets/TECHNICAL_DICTIONARY.md#out-of-domain-ood) rejection on BANKING77 | [OOD](assets/TECHNICAL_DICTIONARY.md#out-of-domain-ood) [F1](assets/TECHNICAL_DICTIONARY.md#f1-score): 94.31% | [OOD](assets/TECHNICAL_DICTIONARY.md#out-of-domain-ood) [F1](assets/TECHNICAL_DICTIONARY.md#f1-score): 87.80% | Label-schema independent safety test |
| [OOD](assets/TECHNICAL_DICTIONARY.md#out-of-domain-ood) rejection on CLINC OOS | [OOD](assets/TECHNICAL_DICTIONARY.md#out-of-domain-ood) [F1](assets/TECHNICAL_DICTIONARY.md#f1-score): 89.16% | [OOD](assets/TECHNICAL_DICTIONARY.md#out-of-domain-ood) [F1](assets/TECHNICAL_DICTIONARY.md#f1-score): 79.23% | Label-schema independent safety test |

---

⚠️ **Why we exclude MASSIVE/SNIPS from headlines:**  
Jane was trained on assistant commands like "turn_on_lights" and "set_reminder".  
MASSIVE/SNIPS use different command names ("light_on", "alarm_set", etc.).  
Only ~50% of their labels could be mapped, so accuracy scores would mislead about Jane's real quality.  
Instead, we show **OOD safety tests** (out-of-domain rejection) which prove this model's smarts on any topic.

---

**Understanding These Benchmarks:**

| Benchmark | What It Tests | What It Means | Example |
|-----------|--------------|---------------|----------|
| **[Latency](assets/TECHNICAL_DICTIONARY.md#latency)** | How fast Jane runs per prediction | Speed is critical for real-time assistants. Under 50ms = excellent; over 200ms = noticeable lag | User says "turn on lights" → model responds in ~25ms (Janus) or ~32ms (v2) |
| **Runtime Reliability** | Can Jane handle 82 multi-turn conversations without crashing? | 0 errors = production-ready; 10+ errors = unstable. Tests real assistant behavior ([clarifications](assets/TECHNICAL_DICTIONARY.md#slot-clarification--clarification-loops), [slot filling](assets/TECHNICAL_DICTIONARY.md#slot-slot-filling), state changes) | Turn 1: "Set alarm" → Turn 45: "Change to 3pm" → Turn 82: Still perfect |
| **[OOD](assets/TECHNICAL_DICTIONARY.md#out-of-domain-ood) Safety (BANKING77)** | Can Jane reject finance questions when trained on home automation? | Tests this model's judgment. ~90% [F1](assets/TECHNICAL_DICTIONARY.md#f1-score) = excellent (rejects what it shouldn't handle). Under 60% = dangerous (would give wrong answers) | User asks "What's my account balance?" → Jane correctly says "I can't help with that" |
| **[OOD](assets/TECHNICAL_DICTIONARY.md#out-of-domain-ood) Safety (CLINC)** | Can Jane reject random real-world off-topic requests? | Similar to BANKING77 but with diverse random questions. Proves this model knows its limits | User asks "What's the capital of France?" → Jane correctly rejects it |

**Bottom Line:** Jane is **SOLID** ✅
- Fast enough for real users (25-31ms per prediction)
- Stable enough for production (0 crashes in 82 turns)
- Safe enough to deploy (87-94% OOD rejection accuracy)

Full detailed report:
- [JaneGPT-v2-Janus/reports/fair_benchmarks.md](JaneGPT-v2-Janus/reports/fair_benchmarks.md)
- [tools/results/fair_benchmarks.md](tools/results/fair_benchmarks.md)

---

## Current Model List

<!-- MODEL-LIST-START -->

| Model | Purpose | Key Highlights | Recommended Entry |
|---|---|---|---|
| [JaneGPT-v2-Janus](JaneGPT-v2-Janus/) | Hierarchical [NLU](assets/TECHNICAL_DICTIONARY.md#nlu-natural-language-understanding) with runtime state | 7.95M params, [domain](assets/TECHNICAL_DICTIONARY.md#domain)+[action](assets/TECHNICAL_DICTIONARY.md#action)+[slot](assets/TECHNICAL_DICTIONARY.md#slot-slot-filling) output, [clarification](assets/TECHNICAL_DICTIONARY.md#slot-clarification--clarification-loops)/pending-slot runtime | [JaneGPT-v2-Janus/README.md](JaneGPT-v2-Janus/README.md) |
| [JaneGPT-v2](JaneGPT-v2/) | Intent classification baseline | 7.8M params, 22 intents, fast lightweight classifier | [JaneGPT-v2/README.md](JaneGPT-v2/README.md) |

<!-- MODEL-LIST-END -->

---

## Repo Layout

- [assets](assets/) - shared visual assets
- [JaneGPT-v2-Janus](JaneGPT-v2-Janus/) - hierarchical NLU + runtime package
- [JaneGPT-v2](JaneGPT-v2/) - intent classifier package

---

## License

Each model folder defines its own license:

- [JaneGPT-v2-Janus License](JaneGPT-v2-Janus/LICENSE)
- [JaneGPT-v2 License](JaneGPT-v2/LICENSE)

---

## Author

Ravindu Senanayake

<p align="right"><a href="#top">Back to top</a></p>
