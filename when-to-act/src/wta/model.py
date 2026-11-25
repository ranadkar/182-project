"""Load the frozen text backbone with verified NF4 CUDA support."""

import importlib.metadata as metadata
import json
import subprocess
import sys

import torch
from transformers import BitsAndBytesConfig, Qwen3_5ForCausalLM, Qwen3_5TextConfig

from .common import CFG, DATA, MODEL, ROOT
from .fetch import sha


def load_model(nf4=True):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires a CUDA GPU with bf16 support")
    cfg = Qwen3_5TextConfig.from_pretrained(MODEL)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        MODEL,
        config=cfg,
        device_map={"": "cuda:0"},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quantization_config=quant if nf4 else None,
        output_loading_info=True,
    )
    if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError(f"Incomplete language backbone: {info}")
    return model


def metadata_for(run):
    return {
        "run": run,
        "config": CFG,
        "python": sys.version,
        "packages": {
            p.metadata["Name"]: p.version
            for p in metadata.distributions()
            if p.metadata["Name"]
        },
        "gpu": torch.cuda.get_device_name(),
        "cuda_runtime": torch.version.cuda,
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "capability": torch.cuda.get_device_capability(),
        "torch_arch_list": torch.cuda.get_arch_list(),
        "template_sha256": sha(MODEL / "chat_template.jinja"),
        "tokenizer_sha256": sha(MODEL / "tokenizer.json"),
        "lock_sha256": sha(ROOT / "uv.lock"),
        "dataset_hashes": json.loads((DATA / "upstream_hashes.json").read_text()),
        "implementation": {
            f.name: sha(f) for f in sorted((ROOT / "src/wta").glob("*.py"))
        },
    }


def run_name(percent=None, seed=None):
    return "baseline" if percent is None else f"non{percent}-seed{seed}"
