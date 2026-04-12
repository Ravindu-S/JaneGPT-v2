<a id="top"></a>

<p align="center">
   <img src="assets/banner.webp" alt="JaneGPT Banner" width="100%" />
</p>

<h1 align="center">JaneGPT Core</h1>

<p align="center"><strong>Assistant-focused model suite for command understanding, hierarchical NLU, and runtime-safe execution</strong></p>

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
   <a href="#current-model-list">Current Model List</a> |
   <a href="#add-a-new-model-in-60-seconds">Add New Model</a>
</p>

---

<p align="center">
  <img src="https://img.shields.io/badge/Model%20Lineup-Choose%20Your%20Engine-0f172a?style=for-the-badge&labelColor=111827&color=1f2937" alt="Model Lineup" />
</p>

<table align="center">
  <tr>
    <td align="center" width="420">
      <a href="JaneGPT-v2-Janus/README.md">
        <img src="https://capsule-render.vercel.app/api?type=rounded&height=56&text=JaneGPT-v2-Janus%20%20%7C%20%20Hierarchical%20NLU%20%2B%20Runtime&fontSize=18&fontAlignY=50&color=0:38bdf8,100:0284c7&fontColor=ffffff&animation=twinkling" alt="JaneGPT-v2-Janus" />
      </a>
      <br />
      <sub><strong>Domain + Action + Slots + Stateful Follow-ups</strong></sub>
    </td>
    <td align="center" width="420">
      <a href="JaneGPT-v2/README.md">
        <img src="https://capsule-render.vercel.app/api?type=rounded&height=56&text=JaneGPT-v2%20%20%7C%20%20Intent%20Classifier%20Baseline&fontSize=18&fontAlignY=50&color=0:4ade80,100:15803d&fontColor=ffffff&animation=fadeIn" alt="JaneGPT-v2" />
      </a>
      <br />
      <sub><strong>Fast 22-intent routing model</strong></sub>
    </td>
  </tr>
</table>

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
| **JaneGPT-v2-Janus** | Real assistant runtime flows | Hierarchical `(domain, action, slots)`, clarifications, pending-slot fill, follow-up state | [Open Janus](JaneGPT-v2-Janus/) |
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

---

## Current Model List

Update only this block when adding new models.

<!-- MODEL-LIST-START -->

| Model | Purpose | Key Highlights | Recommended Entry |
|---|---|---|---|
| [JaneGPT-v2-Janus](JaneGPT-v2-Janus/) | Hierarchical NLU with runtime state | 7.95M params, domain+action+slot output, clarification/pending-slot runtime | [JaneGPT-v2-Janus/README.md](JaneGPT-v2-Janus/README.md) |
| [JaneGPT-v2](JaneGPT-v2/) | Intent classification baseline | 7.8M params, 22 intents, fast lightweight classifier | [JaneGPT-v2/README.md](JaneGPT-v2/README.md) |

<!-- MODEL-LIST-END -->

---

## Repo Layout

- [assets](assets/) - shared visual assets
- [JaneGPT-v2-Janus](JaneGPT-v2-Janus/) - hierarchical NLU + runtime package
- [JaneGPT-v2](JaneGPT-v2/) - intent classifier package

---

## Add A New Model In 60 Seconds

1. Add a new model folder with its own README.
2. Insert one row in `Current Model List` between:
    - `MODEL-LIST-START`
    - `MODEL-LIST-END`
3. Add one quick path under `Quick Paths`.
4. If it is your best/latest model, update `Start Here`.

Template row:

| [Model-Name](Model-Folder/) | One-line purpose | 2-3 strongest highlights | [Model-Folder/README.md](Model-Folder/README.md) |

---

## License

Each model folder can define its own license details:

- [JaneGPT-v2-Janus/LICENSE](JaneGPT-v2-Janus/LICENSE)
- [JaneGPT-v2/LICENSE](JaneGPT-v2/LICENSE)

---

## Author

Ravindu Senanayake

<p align="right"><a href="#top">Back to top</a></p>
