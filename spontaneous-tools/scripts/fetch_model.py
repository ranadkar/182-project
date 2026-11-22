"""Fetch one pinned GGUF conversion. Downloads are resumable and checksum verified."""
import argparse
import hashlib
from pathlib import Path
import subprocess
SHA256 = {
    "olmo": "f20a42d5d72d467c11aebd2de940389c4ad9511febf14957dbcfd3789bedf3bf",
    "nemotron": "be5d9a656a51922f24f1f09a759cebb694e1f5d9728bf0ef9f8c972c5a0b5ef2",
}

MODELS = {
    "olmo": ("bartowski/allenai_Olmo-3-7B-Instruct-GGUF", "0d00999146ae9606a15cf049858a2ccf46fe0706", "allenai_Olmo-3-7B-Instruct-Q4_K_M.gguf"),
    "nemotron": ("nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF", "ba223d14e45525f7fae81db77ea8cabeb2fc6c25", "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()
    repo, rev, name = MODELS[args.model]
    digest = SHA256[args.model]
    dest = Path(".cache") / name
    dest.parent.mkdir(exist_ok=True)
    partial = dest.with_suffix(".part")
    if not dest.exists():
        subprocess.run(["curl", "-fsSL", "--retry", "3", "-C", "-", "-o", str(partial),
                        f"https://huggingface.co/{repo}/resolve/{rev}/{name}"], check=True)
    target = dest if dest.exists() else partial
    with target.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
            raise ValueError(f"checksum mismatch in {target}")
    target.rename(dest)
    print(dest, digest)


if __name__ == "__main__":
    main()
