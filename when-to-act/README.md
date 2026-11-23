# When to act

A UC Berkeley CS 182 experiment on whether demonstrations of restraint change when Qwen3.5-4B calls a tool. The matched comparison varies the non-tool share across 0%, 10%, 25% and 50%. The two primary outcomes measure overcalling and undercalling on the same requests. See [the research report](report.md) for completed runs and [the protocol](configs/protocol.md) for analysis choices.

Eight rank-16 QLoRA adapters completed training on the 12 GB RTX 5070 Ti Laptop. On 72 official free-generation prompts, the unadapted NF4 base made 21 false calls out of 48 and attempted all 24 required calls. At 50% non-tool training, the two seeds made two and four false calls, and both attempted 16 of 24 required calls. The report includes the separate 367-prompt multiple-choice evaluation, exact argument matches, paired intervals and sample limitations. [Behavioral trade-off](figures/official-generate-tradeoff.pdf) · [Per-seed generation results](results/official-generate-seed-results.csv) · [Training and validation curves](figures/loss.pdf)

To rebuild tables and figures from the included predictions, run `source env.sh`, `uv sync --frozen --python /usr/bin/python3`, `uv run --frozen python -m wta.analyze`, and `uv run --frozen python -m wta.report_results`. This path only needs the locked Python environment and saved result files. Run `sha256sum --check results/prediction-and-loss.sha256` to verify the included prediction and loss files.

## Reproduction

Run from this directory on Linux x86-64 with Python 3.12, uv and a CUDA-capable GPU. The lockfile selects the CUDA 12.8 build of PyTorch. Packages, raw data, compiler caches, temporary files and model weights are project-local. Storage must accommodate the 9.3 GB source checkpoint, the Python environment, the package cache and adapters. The upstream data and checkpoint are public. No API key or tool execution is involved.

```bash
source env.sh
uv sync --frozen --python /usr/bin/python3
uv run --frozen python -m wta.fetch
uv run --frozen python -m wta.prepare
uv run --frozen python -m wta.confounds
uv run --frozen python -m wta.check
uv run --frozen python -m wta.hardware
uv run --frozen python -m wta.evaluate baseline dev generate
uv run --frozen python -m wta.evaluate bf16-reference dev generate
uv run --frozen python -m wta.analyze
```

Inspect the generated calls, available names, arguments and text classifications. The development split comes from the preference release. Then run the two endpoint pilots and compare their predictions and losses.

```bash
for pct in 0 50; do
  uv run --frozen python -m wta.train "$pct" 182 || exit 1
  uv run --frozen python -m wta.evaluate "non${pct}-seed182" dev generate || exit 1
done
uv run --frozen python -m wta.analyze
```

After pilot review, complete the same fixed design. Evaluation writes predictions atomically and can resume a partial inference file with the same settings. Restart training from the base if a training run is interrupted.

```bash
for seed in 182 183; do
  for pct in 0 50 10 25; do
    [[ "$seed" == 182 && ( "$pct" == 0 || "$pct" == 50 ) ]] && continue
    uv run --frozen python -m wta.train "$pct" "$seed" || exit 1
    uv run --frozen python -m wta.evaluate "non${pct}-seed${seed}" dev generate || exit 1
  done
done
uv run --frozen python -m wta.freeze
for run in baseline non{0,10,25,50}-seed{182,183}; do
  for protocol in mcq generate; do
    uv run --frozen python -m wta.evaluate "$run" official "$protocol" || exit 1
  done
done
uv run --frozen python -m wta.analyze
uv run --frozen python -m wta.report_results
```

The final analysis command runs offline on existing predictions and loss histories. Each per-example JSONL contains a stable row identifier or UUID, group ID, gold decision and prediction. A preference ID such as `pref-00042` refers to zero-based row 42 in the pinned preference file. Free generation adds raw output, parsed calls and errors. MCQ adds all four candidate scores. Summary JSONs include confusion matrices, denominators, per-seed metrics, paired differences and bootstrap intervals. CSV tables and vector PDF plots can be rebuilt without another model run.

## Supplementary menu-size experiment

The [authored benchmark](data/supplementary-menus.json) contains 18 requests in six domains. The baseline and both seeds of the endpoint adapters each see the same request with zero, two and four extra tools. Inference uses the main experiment’s decoding settings. Its results and paired within-request changes are stored under [results/supplementary](results/supplementary).

```bash
for run in baseline non{0,50}-seed{182,183}; do
  uv run --frozen python -m wta.supplementary "$run" || exit 1
done
uv run --frozen python -m wta.supplementary analyze
```

## Data and model provenance

- [When2Call, NAACL 2025](https://aclanthology.org/2025.naacl-long.174/) by Hayley Ross, Ameya Sunil Mahabaleshwarkar and Yoshi Suhara. The [release](https://huggingface.co/datasets/nvidia/When2Call) is CC BY 4.0. Local manifests identify rows in those files. The fetcher checks the [repository](https://github.com/NVIDIA/When2Call) copy against Hugging Face object identities.
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) from the Qwen team, Apache 2.0. The text backbone and tokenizer come from the same immutable revision. Its official chat template serializes tool calls as nested function and parameter tags.
- [Salesforce xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) documents 60,000 APIGen records containing query, tools and answers. When2Call itself derives training prompts from APIGen. The local training manifests draw exclusively from When2Call chosen preference responses.

