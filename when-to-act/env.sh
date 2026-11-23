# Source from the when-to-act project root.
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export XDG_CACHE_HOME="$PWD/.cache"
export HF_HOME="$PWD/.cache/huggingface"
export TORCH_HOME="$PWD/.cache/torch"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export CUDA_CACHE_PATH="$PWD/.cache/cuda"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export TMPDIR="$PWD/.cache/tmp"
export PYTHONPATH="$PWD/src"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
mkdir -p "$TMPDIR"
