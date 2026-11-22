# Observation-only tool use

**Latest investigation.** The [expanded native-tool evaluation](expanded/README.md) includes 60 paired scenarios, external notifications, typed state transitions and a [matched Qwen3.5-4B LoRA design](expanded/qwen-lora.md). The original pilot and training-data audit are preserved below.


[Research report](report.md) · [Training counts](results/training-summary.json) · [Behavior counts](results/behavior-summary.json) · [Annotated examples with full prefixes](results/examples.jsonl)

Run commands from this directory. The recorded evidence and offline summaries are small. Model weights, the inference runtime and scratch files live under the ignored `.cache/` directory.

## Reproduce the data analysis

Requires Python 3.12 and uv. The recorded run used Python 3.12.3 and uv 0.9.28.

```bash
uv sync --locked
uv run --locked python scripts/analyze.py
uv run --locked python scripts/show.py I-44115448 --through 4
uv run --locked python scripts/behavior_summary.py
```

To resample from the frozen Hugging Face revisions, run the following before the analysis. The sampler uses a matching-revision, untruncated viewer row when available and falls back to Parquet range reads. NVIDIA rows use validated HTTP byte ranges. Some Dolci row groups are large. Download size varies with viewer availability. The unused portions of the multi-GB JSONL files stay remote.

```bash
uv run --locked python scripts/sample.py
uv run --locked python scripts/screen_openings.py
uv run --locked python scripts/pivot_example.py
uv run --locked python scripts/analyze.py
```

| File | Contents |
|---|---|
| [Sample](results/sample.jsonl) | One full upstream row per line, including task messages, tools, available reasoning fields, source identifier, revision and row or byte offset. |
| [Annotations](data/annotations.tsv) | Manual code for every call-bearing assistant turn and a rationale for each trajectory. Indices start at zero. A dash means zero calls. |
| [Turn ledger](results/turns.jsonl) | One record per assistant turn, including turns that end in text, previous role, latest user index, and individual tool names. |
| [Examples](results/examples.jsonl) | Complete messages through six selected decisions, available schemas, source locations and interpretation. The full subsequent outcome is in the sample. |
| [Screen](results/screen.jsonl) and [labels](data/screen-labels.tsv) | Separate discovery sample of opening messages that passed a lexical screen, with full source rows and manual labels. |
| [Pivot example](results/pivot-example.json) | A single illustrative NVIDIA RL training record, including the observed prefix and expected next action. |
| [Scenarios](data/scenarios.json), [Olmo responses](results/olmo.jsonl), [Nemotron responses](results/nemotron.jsonl) | Paired observations and per-trial prompts, raw generations, stop reasons, simulated outcomes and explicit-check targets. |

## Reproduce inference

On Ubuntu 24.04 x86-64 with an existing CUDA-capable driver, these commands fetch the pinned llama.cpp b12600 CUDA 12.8 runtime into the project, verify the archives, and fetch and verify two Q4_K_M GGUFs. The weights total 7.31 GB in decimal units. No system installation is required. The pilot uses one local server and one model at a time.

```bash
bash scripts/install_runtime.sh
uv run --locked python scripts/fetch_model.py nemotron
uv run --locked python scripts/fetch_model.py olmo
bash scripts/run_pilot.sh nemotron .cache/replication/nemotron.jsonl
bash scripts/run_pilot.sh olmo .cache/replication/olmo.jsonl
uv run --locked python scripts/behavior_summary.py --folder .cache/replication
```

The launcher uses the raw completion endpoint and loopback port 8182. The harness renders the prompt, and the server starts with its built-in ChatML template to avoid an unused native-parser initialization error on Olmo. Set `PILOT_PORT` to select another. It requires a new output path so a rerun preserves the recorded observations. To use a compatible existing llama.cpp server on other hardware, run `uv run --locked python scripts/evaluate.py --help`. The scorer requires the pinned GGUF filename. [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b12600)

## Source data and attribution

The Dolci sample is from the Olmo team at the Allen Institute for AI, released under ODC-BY for research and educational use. The Agentic and Pivot samples are from NVIDIA, licensed under CC BY 4.0. Extracted rows preserve source content. Sampling keys, annotations and interpretations were added for this study. NVIDIA also credits Glaive for its tool-calling source. The dataset licenses and model licenses are linked in the upstream cards. [Dolci card](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT-Tool-Use/blob/dc042846f0f2de0f15eedae3d6ced04223ed47eb/README.md) · [NVIDIA Agentic card](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1/blob/650d590978ca35c8f1ecea2faf136e5fac421b62/README.md) · [NVIDIA Pivot card](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1/blob/bfc7a4dd55ec1424cc3761ceacdd85a5743b4f4e/README.md)

The completed Qwen3.5-4B controlled fine-tuning experiment includes the matched datasets, eight trained adapters, first-turn outputs, uncertainty estimates, and reproduction commands. [Qwen experiment report](qwen/report.md)
