# Reproducing the experiment

## Frozen predictions

The [CPU workflow](../README.md#reproduce-the-results) rebuilds the 300-prompt generation table, paired composition effects, matched MCQ discordances and direct-answer family analysis. Each analysis directory under [results](../when-to-act/results/) contains a readable `tables.md`, a `metrics.csv` and an unrounded `summary.json`. The generation, matched and control plots are written under `when-to-act/figures/{expanded-generation,matched-protocols,direct-answer-controls}/`. The same command rebuilds the six data tables in [tables.json](../when-to-act/paper/tables.json) and the five [manuscript figures](../when-to-act/paper/figures/). It checks their SHA-256 digests. To compile the PDF with Typst 0.14.2 and the bundled fonts, run from the repository root

```bash
typst compile --creation-timestamp 0 --ignore-system-fonts --font-path when-to-act/paper/fonts when-to-act/paper/main.typ when-to-act/paper/paper.pdf
```

The script checks the original prediction and loss digests, then compares regenerated numerical outputs with the frozen digests. The 2,000 percentile-bootstrap draws use the same multiplicities across models and protocols. Each draw resamples 272 source-question clusters, or 36 entire families for the control. Rates use the denominator in that draw, and adapter means give equal weight to the two fits. Intervals are conditional on those fits. PDF rendering can vary with platform font libraries. [Original records](../when-to-act/results/prediction-and-loss.sha256) · [Evaluation artifacts](../when-to-act/data/followup/outputs.sha256) · [Analysis protocol](../when-to-act/data/followup/protocol.md)

## GPU environment and data

Use Bash on Linux x86-64 with Git, curl, uv and a CUDA GPU supporting bf16 and the locked NF4 kernels. The recorded hardware was an RTX 5070 Ti Laptop GPU with 12 GB VRAM. The eight training and validation loops took 87.6 minutes together, excluding data preparation and inference. Allow disk space for the 9.3 GB base model plus the environment, package cache and outputs. Runtime and fitted weights can vary across devices, drivers and numerical implementations.

The original experiment has its own Python 3.12 project. Its lockfile hash is part of the evaluation freeze. It pins PyTorch 2.10.0+cu128 (CUDA 12.8), Transformers 5.4.0, PEFT 0.18.1 and bitsandbytes 0.49.2. The root CPU environment is independent. [GPU lockfile](../when-to-act/uv.lock) · [Hardware record](../when-to-act/results/hardware.json) · [Frozen settings](../when-to-act/configs/experiment.json)

Use a separate checkout for fitting or inference. The original scripts write manifests, run metadata and predictions in place. From the repository root, start a new worktree and put the shipped results aside there so completed files cannot cause inference to be skipped.

```bash
set -e
git worktree add --detach ../182-gpu HEAD
cd ../182-gpu/when-to-act
source env.sh
uv sync --frozen
mv results .cache/published-results
mkdir results
cp .cache/published-results/freeze.json results/
.venv/bin/python -m wta.fetch
.venv/bin/python -m wta.prepare
.venv/bin/python -m wta.check
```

The fetcher checks NVIDIA's Git data files against the pinned Hugging Face objects, and verifies the tokenizer, template and weight shards. Training uses only chosen preference responses, indexed by zero-based row IDs in the pinned file. Preparation excludes ambiguous targets, malformed calls, empty menus, overlength rows and near-duplicate training/test questions. It reserves a 48-example development set, then creates nested replacements matched on tool count, preferring identical or overlapping menus. The four 1,000-row manifests share a slot order. The split seed is 1822026. [Manifests and hashes](../when-to-act/data/) · [Filtering code](../when-to-act/src/wta/prepare.py) · [Data audit](../when-to-act/results/data_audit.json)

| Input | Frozen revision |
|---|---|
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call/tree/0582f7749df63a96fdc3070932e83e72396ace53) | `0582f7749df63a96fdc3070932e83e72396ace53` |
| [NVIDIA/When2Call](https://github.com/NVIDIA/When2Call/tree/ecc8d42388e91ab37e7e737d48e16e8ecea3d1dc) | `ecc8d42388e91ab37e7e737d48e16e8ecea3d1dc` |

## Use published weights

Request access on the [adapter repository](https://huggingface.co/rajnadkarni/when-to-act-qwen35-adapters), then make a read token for that account available through `HF_TOKEN` in this shell. In the GPU worktree above, this downloads and loads one adapter. The original PEFT configs record a training-machine path, so load the pinned backbone explicitly.

```bash
adapter_repo=rajnadkarni/when-to-act-qwen35-adapters
adapter_rev=f20e7e53de1a76c2a22eca7770e31079be4a98be
.venv/bin/hf download "$adapter_repo" --revision "$adapter_rev" \
  --include 'non50-seed182/adapter_*' --local-dir checkpoints
.venv/bin/python - <<'PY'
from peft import PeftModel
from wta.model import load_model

model = PeftModel.from_pretrained(load_model(), "checkpoints/non50-seed182").eval()
print(model.is_loaded_in_4bit)
PY
```

The folders are `non0-seed182`, `non0-seed183`, `non10-seed182`, `non10-seed183`, `non25-seed182`, `non25-seed183`, `non50-seed182` and `non50-seed183`. Each contains `adapter_config.json` and `adapter_model.safetensors`. The [checkpoint audit](../when-to-act/results/expanded-generation/audit.json) records both SHA-256 digests for each fit.

For the **full evaluation from published weights**, fetch all eight and carry the corresponding training records into the new results directory. To train new weights instead, use the next section.

```bash
.venv/bin/hf download "$adapter_repo" --revision "$adapter_rev" \
  --include 'non*/adapter_*' --local-dir checkpoints
for run in non{0,10,25,50}-seed{182,183}; do
  mkdir -p "results/$run"
  cp ".cache/published-results/$run/"{training.json,loss.jsonl} "results/$run/"
done
```

## Train new weights

The text backbone uses double-quantized 4-bit NF4 with bf16 compute and SDPA. LoRA targets `q_proj`, `v_proj` and `in_proj_qkv` with rank 16, alpha 32 and zero dropout. Each run shuffles its mixture once. The trainer takes one sequence per microbatch (maximum 1,024 tokens), accumulates eight gradients, and makes 125 fused AdamW updates. The peak learning rate is 0.0001 with five warmup steps and cosine decay, zero weight decay and gradient norm capped at 1. Loss covers assistant tokens including the end-of-turn token. The final update supplies the adapter. Run one combination or the full grid below. [Trainer](../when-to-act/src/wta/train.py) · [Example training order and metadata](../when-to-act/results/non50-seed182/training.json)

```bash
for seed in 182 183; do
  for pct in 0 50 10 25; do
    .venv/bin/python -m wta.train "$pct" "$seed"
  done
done
```

## Rerun inference and analysis

Use one complete set of eight adapters from either path above. The original procedure first scored a 72-prompt generation sample and a 367-prompt MCQ sample. The final evaluation reuses 72 serialized generation prompts and 27 identical MCQ inputs per model. Recreate those initial predictions in the **same environment** as the final pass. The freeze and reuse checks compare the configuration, source hashes, GPU, packages, tokenizer, serialized inputs and weights. The shipped historical metadata describes the original machine. The new worktree records the current run's metadata.

```bash
runs=(baseline non{0,10,25,50}-seed{182,183})
for run in "${runs[@]}"; do
  .venv/bin/python -m wta.evaluate "$run" dev generate
done
.venv/bin/python -m wta.freeze
for run in "${runs[@]}"; do
  for protocol in generate mcq; do
    .venv/bin/python -m wta.evaluate "$run" official "$protocol"
  done
done

.venv/bin/python -m wta.followup prepare
.venv/bin/python -m wta.followup_controls prepare
for protocol in generate mcq control; do
  for run in "${runs[@]}"; do
    .venv/bin/python -m wta.followup "$protocol" "$run"
  done
done
```

The main analysis takes every judge UUID and matches released candidate order, question and menu. The no-thinking Qwen template serializes the prompt. Generation is greedy with left padding, batch size 4 (reduced on OOM), and a 160-new-token cap or end-of-turn stop. The local first-act classifier counts malformed attempts as calls. Strict exact recall requires a valid full call list equal to the reference. All 100 required prompts stay in the denominator, including four invalid references. MCQ scores the four native continuations without EOS and chooses the largest log likelihood per UTF-8 byte, with ties resolved by candidate order. Raw-sum scores are also saved. The authored control pairs a required lookup, a missing-identifier request and a supplied-fact request in each family. No tools are executed. [Classifier](../when-to-act/src/wta/decisions.py) · [Scoring code](../when-to-act/src/wta/evaluate.py) · [Control design](../when-to-act/data/followup/direct-design.tsv)

Compute estimates for the new predictions with the same analysis modules. The checksum comparison in the root reviewer command applies to the published records. An inference replay uses the same weights but can vary with batch padding or numerical implementation. Repeating training yields new fitted models.

```bash
for phase in expanded matched; do
  for module in stats tables figures; do
    .venv/bin/python -m "wta.followup_${module}" "$phase"
  done
done
.venv/bin/python -m wta.followup_control_stats
```

For the published runs, 54 saved blinded judgments document the first-act spot check. With the pinned data and tokenizer available, `wta.followup_verify` also replays prompt overlap, classifier decisions and candidate-score normalization. [Review and disagreements](../when-to-act/results/matched-protocols/review.json) · [Score replay](../when-to-act/src/wta/followup_verify.py)
