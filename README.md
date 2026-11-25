# The Cost of Restraint

**How Fine-Tuning Mixtures Shape Tool Use in Qwen3.5-4B** · UC Berkeley CS 182

How often does a model still call a tool when it should answer in text, and how many legitimate calls does it miss as it learns to abstain? We fine-tuned Qwen3.5-4B with rank-16 QLoRA. Each mixture contains 1,000 demonstrations. We replaced 0%, 10%, 25% or 50% of the tool-call targets with clarification or inability responses and trained two adapters per mixture.

## Results

Greedy generation on all 300 When2Call judge prompts. False calls are measured on the 200 text-response cases. Required-call recall counts attempts on the 100 tool-required cases. Exact recall also requires valid function names and arguments matching the reference. Adapter rows average seeds 182 and 183.

| Non-tool share | False calls | Required-call recall | Exact recall |
|---|---:|---:|---:|
| Unmodified base (NF4) | 47.5% | 99.0% | 61.0% |
| 0% | 91.5% | 100.0% | 64.5% |
| 10% | 37.8% | 96.0% | 64.5% |
| 25% | 18.8% | 83.0% | 58.5% |
| 50% | 10.0% | 67.0% | 50.5% |

On the same required prompts, the 50% adapters selected the supplied tool call 99.0% of the time but generated a call only 67.0% of the time. In a separate exploratory control with 36 authored lookup families, they asked for clarification in every missing-identifier case yet called on 98.6% of requests that already supplied the answer. Restraint on the official benchmark transferred poorly to these directly answerable tasks.

## Paper and models

The repository includes a [manuscript PDF](when-to-act/paper/paper.pdf) and [Typst source](when-to-act/paper/main.typ). The eight [QLoRA adapters on Hugging Face](https://huggingface.co/rajnadkarni/when-to-act-qwen35-adapters) require a signed-in access request. Hugging Face grants access automatically. The predictions are included in Git.

## Reproduce the results

From a shell with Git and [uv](https://github.com/astral-sh/uv), run

```bash
git clone https://github.com/ranadkar/182-project.git
cd 182-project
uv sync --locked
uv run --locked python scripts/reproduce.py
```

The CPU workflow was tested on Linux x86-64 with Python 3.12. After setup, analysis runs offline on CPU with no model weights or credentials. The script recomputes 2,000 paired bootstrap draws, regenerates the manuscript tables and five SVG figures, and verifies them against the frozen SHA-256 digests. Browse the [generation results](when-to-act/results/expanded-generation/tables.md), [matched MCQ comparison](when-to-act/results/matched-protocols/tables.md) and [direct-answer controls](when-to-act/results/direct-answer-controls/tables.md). Regenerated PDFs are under `when-to-act/figures/`.

## Training and inference

The [reproduction guide](docs/reproducibility.md) covers data filtering, the frozen CUDA environment, loading an adapter, retraining and evaluating all three protocols. Under `when-to-act/`, [configs](when-to-act/configs/) fixes the experiment, [data](when-to-act/data/) holds manifests, [src/wta](when-to-act/src/wta/) implements it and [results](when-to-act/results/) contains predictions, run metadata and statistical estimates.

## Data and attribution

We used chosen responses from the When2Call preference split and the released judge and MCQ test records. The dataset is CC BY 4.0. This repository supplies filtered manifests, normalized evaluation inputs and derived predictions. [Ross et al., NAACL 2025](https://aclanthology.org/2025.naacl-long.174/) · [NVIDIA dataset](https://huggingface.co/datasets/nvidia/When2Call) · [upstream code](https://github.com/NVIDIA/When2Call)

The Qwen team released the base checkpoint under Apache 2.0. TeX Gyre Termes carries its own font license. A separate license for this project's original code has yet to be selected. [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) · [font notices](when-to-act/paper/fonts/)
