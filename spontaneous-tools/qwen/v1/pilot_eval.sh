#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
py="$(cd ../.venv/bin && pwd)/python"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
"$py" inference.py base development --precheck
for model in pilot-{explicit,factual,neutral,continuation}-7741; do
    "$py" inference.py "$model" development
done
