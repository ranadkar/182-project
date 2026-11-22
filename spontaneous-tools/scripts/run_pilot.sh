#!/usr/bin/env bash
# Start a fresh server for one model. Pass a new output path for a replication.
set -euo pipefail
model=${1:?supply olmo or nemotron}
output=${2:?supply output JSONL path}
[[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 1; }
case "$model" in
    olmo) weights=allenai_Olmo-3-7B-Instruct-Q4_K_M.gguf ;;
    nemotron) weights=NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf ;;
    *) echo "Unknown model: $model" >&2; exit 1 ;;
esac
runtime="$PWD/.cache/llama-b12600"
cuda_runtime="$PWD/.cache/cudart-llama-b12600-bin-ubuntu-cuda-12.8-x64"
export LD_LIBRARY_PATH="$cuda_runtime:$runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
port=${PILOT_PORT:-8182}
endpoint="http://127.0.0.1:$port"
log=".cache/$model-server.log"
"$runtime/llama-server" --offline --no-webui --no-jinja --chat-template chatml --host 127.0.0.1 --port "$port" \
    -m ".cache/$weights" -ngl 99 -t 8 -c 2048 -np 1 >"$log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
for ((attempt=0; attempt<120; attempt++)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then cat "$log" >&2; exit 1; fi
    if curl -fsS "$endpoint/health" >/dev/null 2>&1; then break; fi
    sleep 1
done
uv run --locked python scripts/evaluate.py "$model" --endpoint "$endpoint" --output "$output"
