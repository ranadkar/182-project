#!/usr/bin/env bash
# Project-local Ubuntu x86-64 CUDA 12.8 build. Run from the research directory.
set -euo pipefail
mkdir -p .cache
cd .cache
release=b12600
base_url="https://github.com/ggml-org/llama.cpp/releases/download/$release"
fetch() {
    local filename=$1 archive=$2 digest=$3
    if [[ ! -f "$archive" ]]; then
        curl -fL --retry 3 -C - -o "$archive.part" "$base_url/$filename"
        mv "$archive.part" "$archive"
    fi
    echo "$digest  $archive" | sha256sum --check
    tar -xzf "$archive"
}
fetch "llama-$release-bin-ubuntu-cuda-12.8-x64.tar.gz" llama.tgz 6b2bbca74c951e03e2cb45771410ac58590b6677257ae78e2c8a896bb0a3e1b2
fetch "cudart-llama-$release-bin-ubuntu-cuda-12.8-x64.tar.gz" cudart.tgz 971775211e47dad69f8143eab527893abaacdd69b91005ff3621c72692cfe199
