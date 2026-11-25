"""Regenerate the CS 182 evaluation results from the frozen prediction files."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1] / "when-to-act"


def verify(manifest):
    for line in (PROJECT / manifest).read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = PROJECT / name
        # PDF rendering can depend on the platform's font libraries.
        if path.suffix == ".pdf":
            if not path.read_bytes().startswith(b"%PDF-"):
                raise ValueError(f"Missing or invalid figure: {name}")
            continue
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: {name} (expected {expected}, got {actual})")


def main():
    env = dict(
        os.environ,
        PYTHONPATH=str(PROJECT / "src"),
        MPLCONFIGDIR=str(PROJECT.parent / ".cache/matplotlib"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
    )

    def run(*args):
        subprocess.run([sys.executable, *args], cwd=PROJECT, env=env, check=True)

    verify("results/prediction-and-loss.sha256")
    for phase in ("expanded", "matched"):
        print(f"Rebuilding {phase} estimates and figures...", flush=True)
        for module in ("stats", "tables", "figures"):
            run("-m", f"wta.followup_{module}", phase)
    print("Rebuilding the authored control...", flush=True)
    run("-m", "wta.followup_control_stats")
    run("paper/build_figures.py")
    verify("data/followup/outputs.sha256")

    tables = PROJECT / "results/expanded-generation/tables.md"
    print(tables.read_text().partition("### Per-model counts")[0].strip())
    print("\nAll frozen numerical summaries, intervals, manuscript tables and SVG figures match.")
    print("Tables: when-to-act/results/   Manuscript figures: when-to-act/paper/figures/")


if __name__ == "__main__":
    main()
