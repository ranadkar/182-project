#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
py=.venv/bin/python
"$py" generate.py
"$py" evaluate.py base calibration evaluation competence
for arm in continuation explicit; do
    if [[ ! -f adapters/pilot-${arm}-7741/run.json ]]; then
        "$py" train.py "$arm" 7741 --pilot
    fi
    "$py" evaluate.py "pilot-${arm}-7741" calibration pilotcheck
done
for seed in 18235 60921; do
    for arm in continuation explicit ack; do
        if [[ ! -f adapters/main-${arm}-${seed}/run.json ]]; then
            "$py" train.py "$arm" "$seed"
        fi
    done
done
"$py" freeze.py
for seed in 18235 60921; do
    for arm in continuation explicit ack; do
        "$py" evaluate.py "main-${arm}-${seed}" calibration evaluation competence
    done
done
for model in base main-{continuation,explicit,ack}-{18235,60921}; do
    "$py" evaluate.py "$model" reserve
done
"$py" analyze.py
"$py" verify.py
