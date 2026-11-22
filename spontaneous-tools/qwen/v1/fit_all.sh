#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
py=../.venv/bin/python
"$py" training_data.py
for arm in explicit factual neutral continuation; do
    [[ -f adapters/pilot-$arm-7741/run.json ]] || "$py" fit.py "$arm" 7741 --pilot
done
for seed in 18235 60921 72467; do
    for arm in explicit factual neutral continuation; do
        [[ -f adapters/main-$arm-$seed/run.json ]] || "$py" fit.py "$arm" "$seed"
    done
done
