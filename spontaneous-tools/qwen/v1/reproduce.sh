#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
py="$(cd ../.venv/bin && pwd)/python"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
"$py" training_data.py
"$py" prepare.py
"$py" checks.py
for arm in explicit factual neutral continuation; do
    [[ -f adapters/pilot-$arm-7741/run.json ]] || "$py" fit.py "$arm" 7741 --pilot
done
./pilot_eval.sh
"$py" amendment.py
./fit_all.sh
./score_all.sh
"$py" mechanisms.py
"$py" tables.py --update-report
"$py" bridge.py
