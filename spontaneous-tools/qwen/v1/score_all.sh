#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
py="$(cd ../.venv/bin && pwd)/python"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
"$py" checks.py
"$py" seal.py
models=(base)
for seed in 18235 60921 72467; do
    for arm in explicit factual neutral continuation; do models+=("main-$arm-$seed"); done
done
for model in "${models[@]}"; do "$py" inference.py "$model" development final; done
for model in "${models[@]}"; do "$py" inference.py "$model" replication; done
"$py" verify.py
"$py" analyze.py
"$py" plots.py

"$py" leakage.py
"$py" examples.py
"$py" cache_check.py
