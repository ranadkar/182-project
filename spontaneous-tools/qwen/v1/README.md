# Qwen four-condition study for CS 182

[Report](report.md) · [Hypotheses and frozen design](design.md) · [Exact traces](examples.md) · [First-report notes](paper_notes.md)

Use the existing parent Qwen project and its pinned model, BF16 runtime, local Python environment and dependency lock. From the `spontaneous-tools` directory:

```bash
cd qwen
uv sync --frozen --python 3.14
uv run --frozen python fetch.py
# On a fresh checkout only, install the local runtime and pinned converter.
(cd .. && bash scripts/install_runtime.sh)
git clone --filter=blob:none --depth 1 --branch b12600 --sparse \
  https://github.com/ggml-org/llama.cpp.git ../.cache/qwen-llama-source
git -C ../.cache/qwen-llama-source sparse-checkout set gguf-py conversion
cd v1
./reproduce.sh
```

The existing workspace already contains these local runtime files. The full pipeline skips finished fits and resumes complete JSONL records by their frozen trial keys. For a clean independent rerun use a fresh copy of the source with empty `adapters/`, `results/`, `figures/`, `raw/`, and `diagnostics/`, and omit the prior run's generated `data/final_lock.json`. The source suite is `data/cases.txt`. All generated prompts, states and paired training histories are retained. The earlier study's scripts, weights and results live one directory above. No account, privileged installation or remote inference service is needed.

### Reanalyze saved outputs

From `qwen/`:

```bash
OPENBLAS_NUM_THREADS=1 uv run --frozen python v1/analyze.py
uv run --frozen python v1/plots.py
uv run --frozen python v1/verify.py
OPENBLAS_NUM_THREADS=1 uv run --frozen python v1/mechanisms.py
uv run --frozen python v1/examples.py
uv run --frozen python v1/tables.py --update-report
```

`results/trials.csv` contains one row per actual model generation at the 96-token endpoint. When a new-suite substantive generation hits the cap, fields ending in `_256` give the effect of continuing the same output. Raw JSONL files include generated token IDs, complete response text, score, intermediate states, stop reason, timing, prompt hash and adapter hash. Join a raw trial's `key` to `data/evaluation_prompts.jsonl` for the exact prompt. Pilot records and the initial base interface check use `data/prompts.jsonl`. `diagnostics/` retains that separate original base check. No entries for the December holdback are in either prompt bank. Its authored definitions are retained for the next study. The two prompt versions and the base development precheck have separate provenance.

`results/by_seed.csv` contains counts and denominators. `results/estimates.csv` contains rates and paired differences as proportions, with lower and upper bounds. A blank reference denotes a single condition. The `analysis` column distinguishes cell estimates, within-scenario evidence differences and delivery differences. `domain_contrasts.csv` and `discordances.csv` break down the primary comparisons. `fits.csv` and `loss_curves.csv` contain fitting resources and loss measurements. The executable analysis fixes its 10,000 bootstrap draws with NumPy seed 182910. The inherited training configuration includes a dormant bootstrap seed of 182. The exploratory tables in `paired_outcome_transitions.csv`, `operation_counts.csv` and `common_competence.csv` count paired outcome switches, operation-level behavior and performance on cases where every model correctly executed the requested action. `coupled_changes.csv` pairs the changes in spontaneous target matching with changes in open-task accuracy on the very same case and seed. Conditioning on an outcome affected by training is descriptive. `protocol_diagnostics.csv` reports parser errors and length stops at both endpoints.

### Scoring conventions

A whole predicted turn executes atomically. The parser recognizes the official XML-like Qwen function envelope. It validates each operation's parameter names, enum membership, integer syntax and bounds, and ledger constraints. A read has no state effect. A valid write may leave the ledger unchanged, change the intended entity, or change the wrong entity. A redundant write means at least one executed write left its immediate pre-call state unchanged. Two exchanges that cancel each other leave no net change but neither exchange is individually redundant. Target matching requires a net change and exact equality of the complete final ledger to the registered target. On ordinary, ambiguous and conflicting cells, that target belongs to the paired discrepancy; use `changed` to measure mutation on the control. `abstention` includes explicit pass, ordinary text and an empty response. An invalid turn is separate. Recognizable write attempts require a known mutation name in a recognizable invalid call or a valid write. A prose recommendation alone never executes.

The software reuses the native renderer, strict decoder, LoRA optimizer and GPU runtime from the earlier study. [Parent implementation](../README.md) · [Simulator and prompts](suite.py) · [Input freeze](data/freeze.json) · [Development amendment](data/amendment.json)


### Exploratory comparison with the original adapters

After the registered evaluation we ran both generations of saved explicit and continuation adapters on just the 108 external discrepancy prompts. The two shared seed IDs, fixed alphabetical scenario order, cold KV cache, ten slots, greedy decoding and 96-token endpoint are identical in this replay. All 864 supplemental responses are in `diagnostics/bridge/`. The crossover joins these responses to the original 60-case results for the same four adapters in each fitting protocol. It uses whole-domain and paired-seed intervals within each benchmark and keeps these exploratory trials separate from the confirmatory CSV. From `qwen/` run:

```bash
OPENBLAS_NUM_THREADS=1 uv run --frozen python v1/bridge.py
OPENBLAS_NUM_THREADS=1 uv run --frozen python v1/bridge.py --analyze-only
```

The second command reanalyzes the saved supplemental outputs on CPU. The required original adapters and their earlier evaluation JSONL files are in the parent study. [Supplemental estimates](results/bridge_estimates.csv) · [Crossover trial table](results/bridge_trials.csv)
