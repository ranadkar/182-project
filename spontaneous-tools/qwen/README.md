# Qwen training and evaluation

The four-condition CS 182 follow-up has its own reproduction entry point, adapters, independently authored benchmark and results. [V1 study](v1/README.md)

Run commands from `spontaneous-tools/qwen` on Linux x86-64 with a CUDA-capable GPU. The full study used one 12 GB laptop GPU. The software lock is independent of the earlier study's environment. Model weights, converter source and inference binaries live under the parent project's ignored `.cache/`. The PEFT adapters and generated records are under this directory.

```bash
uv sync --frozen --python 3.14
uv run --frozen python fetch.py
(cd .. && bash scripts/install_runtime.sh)
git clone --filter=blob:none --depth 1 --branch b12600 --sparse \
  https://github.com/ggml-org/llama.cpp.git ../.cache/qwen-llama-source
git -C ../.cache/qwen-llama-source sparse-checkout set gguf-py conversion
git -C ../.cache/qwen-llama-source rev-parse HEAD
./pipeline.sh
```

The expected converter commit is `29b52d398b294509746ec0a3f28948a3107c0213`. Existing installations can skip the runtime installer and clone. The fetcher compares both official LFS SHA-256 digests before accepting existing local weights. Reuse a trusted cached snapshot by linking the two weight shards into `../.cache/qwen-851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/` before fetching the smaller files. Never write through those links.

For an independent rerun use a fresh copy with empty `adapters/` and `results/` directories. The pipeline reuses finished runs and resumes interrupted evaluation logs. To reproduce only the analysis from the saved outputs, run the command below. The CSV files include cell counts, domain and case bootstrap contrasts, crossed domain/seed intervals for secondary outcomes, and gains and losses on paired cases.

```bash
uv run --frozen python analyze.py
```

To inspect a single fitting run and its positive controls on a fresh copy:

```bash
uv run --frozen python generate.py
uv run --frozen python train.py continuation 7741 --pilot
uv run --frozen python evaluate.py pilot-continuation-7741 calibration pilotcheck
```

[Study protocol](design.md) · [Budget and seeds](config.json) · [Training-data audit](data/audit.json) · [Findings](report.md)
