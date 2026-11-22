# Native tool-use experiment

[Findings in the project report](../report.md#expanded-behavioral-evaluation) · [Frozen design and exact role manipulation](design.md) · [Causal fine-tuning design](qwen-lora.md)

Run from the `spontaneous-tools/` directory with the existing project-local uv environment. The cached checkpoints and GPU runtime are the same as the pilot. For a new machine, follow the [pilot installation instructions](../README.md#reproduce-inference) for the two GGUFs and local llama.cpp b12600 runtime.

```bash
uv sync --locked
uv run --locked python expanded/scenarios.py
uv run --locked python expanded/test_native.py
uv run --locked python expanded/analyze.py
```

To reproduce inference into a new directory, run the following sequentially. The main runner checks the calibration threshold and fingerprints of the renderer, manifest, model and run settings. It can resume a log of complete JSON lines. Each request contains a newly rendered full prompt with a fresh initial state. No conversation history is carried forward. The bridge uses the original pilot stimulus files, tool enums in place of arbitrary strings, and the same native renderer as the expanded experiment.

```bash
uv run --locked python expanded/run.py calibration --folder .cache/expanded-replication
uv run --locked python expanded/run.py main --folder .cache/expanded-replication
uv run --locked python expanded/bridge.py --folder .cache/expanded-replication
uv run --locked python expanded/analyze.py --folder .cache/expanded-replication
```

| Evidence | Contents |
|---|---|
| [Scenario bank](scenario_bank.txt) and [generated manifest](scenarios.json) | Each row defines a distinct setting, typed action, hidden initial state, target and two factual observations. The manifest includes the explicit-action control and counterbalanced menu order. |
| [Native protocol and simulator](native.py) | Olmo tagged Python calls and Nemotron tagged function/parameter blocks. Read returns the typed setting. Set accepts the declared enum, integer range or boolean. Multiple calls are ordered and an invalid batch has no effect. |
| [Olmo calibration](results/olmo-calibration.jsonl) and [Nemotron calibration](results/nemotron-calibration.jsonl) | Independent read, write and no-action checks run before main inference. |
| [Olmo trials](results/olmo-main.jsonl) and [Nemotron trials](results/nemotron-main.jsonl) | Full rendered prompts, tool order, raw output, generated token IDs, decoded arguments, simulator transitions, EOS/limit status, checksums and build. Every model has 240 factual trials and 60 explicit controls. |
| [Cell rates](results/rates.csv) | Counts, denominators, percentages and paired-scenario bootstrap percentile intervals. |
| [Contrasts](results/contrasts.csv) | Signed percentage-point differences with intervals and counts of positive and negative within-scenario contrasts. For two-cell comparisons these are the discordant pairs. |
| [Diagnostics](results/diagnostics.json) | Calibration success, explicit-control failures, length stops, multi-call decisions and parse rejections. |
| [Olmo bridge](results/olmo-bridge.jsonl), [Nemotron bridge](results/nemotron-bridge.jsonl), [paired comparison](results/bridge.csv) | Exact pilot wording replayed on the native harness. |
| [Review of rejected outputs](attempt_audit.tsv) | Single-reviewer distinction between an attempted command and an explanation containing tool syntax. The raw parser outcome remains in each generation record. Offline rates use the reviewed labels for ordinary replies and malformed attempts. The invalid_candidates column retains the original flag. A new rejection with different output bytes remains flagged and is listed for review in diagnostics. |

The inference build is restricted to loopback, optional tool selection and one assistant turn. The environment is a local Python simulation. An explicit no-action tool is a valid executed interface call. The operational-call metric counts reads or writes. The primary change metric compares the final value with the initial value. For ordinary controls, reaching the discrepancy target is reported as a placebo direction.

**Template provenance.** The two templates are unmodified files from the pinned model repositories. Olmo's file has SHA-256 `f5186d42d99c8a0445d37fd8a6c7ccf07fe3e24a29ce622d8bd245da9507b12b`. Nemotron's file has SHA-256 `ab7813c3abdd9cb655905a410728b26c7884eca45ddfab8d9f931553485a7862`. The GGUF sources and SHA-256 checksums are in the project's existing model fetcher. [Ai2 Olmo template and license](https://huggingface.co/allenai/Olmo-3-7B-Instruct/tree/6e5971d9eba42665f5bd5a0fcf047f299ce1dccc) · [NVIDIA Nemotron template and license](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/tree/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f) · [GGUF pins](../scripts/fetch_model.py)
