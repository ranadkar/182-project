"""Fetch pinned upstream artifacts into project-local storage; verify all downloaded bytes."""

import hashlib
import json
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT / "configs/experiment.json").read_text())
MODEL = ROOT / ".cache/model"
UPSTREAM = ROOT / ".cache/upstream/When2Call"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path, entry):
    digest = sha(path)
    if "lfs" in entry:
        matches = digest == entry["lfs"]["oid"]
    else:
        blob = subprocess.check_output(
            ["git", "hash-object", str(path)], text=True
        ).strip()
        matches = blob == entry["oid"]
    if path.stat().st_size != entry["size"] or not matches:
        raise ValueError(f"Upstream object hash mismatch: {path}")
    return digest


def transfer(path, url, size):
    partial = path.with_suffix(path.suffix + ".part")
    if size < 1 << 30:
        subprocess.run(
            ["curl", "-fLsS", "--retry", "5", "-C", "-", "-o", str(partial), url],
            check=True,
        )
    else:
        start = partial.stat().st_size if partial.exists() else 0
        chunk_size = 64 << 20

        def chunk(begin):
            end = min(size, begin + chunk_size) - 1
            piece = partial.with_suffix(f".{begin}.chunk")
            if not piece.exists() or piece.stat().st_size != end - begin + 1:
                subprocess.run(
                    [
                        "curl",
                        "-fLsS",
                        "--retry",
                        "5",
                        "--range",
                        f"{begin}-{end}",
                        "-o",
                        str(piece),
                        url,
                    ],
                    check=True,
                )
            if piece.stat().st_size != end - begin + 1:
                raise ValueError(f"Bad range {piece}")
            return piece

        import shutil

        with ThreadPoolExecutor(8) as pool, partial.open("ab") as dest:
            for piece in pool.map(chunk, range(start, size, chunk_size)):
                with piece.open("rb") as src:
                    shutil.copyfileobj(src, dest)
                dest.flush()
                piece.unlink()
    partial.replace(path)


def main():
    if not UPSTREAM.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/NVIDIA/When2Call", str(UPSTREAM)],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(UPSTREAM), "checkout", "--detach", CFG["repo_revision"]],
        check=True,
    )
    hashes = {}
    for split in ["train", "test"]:
        api = f"https://huggingface.co/api/datasets/nvidia/When2Call/tree/{CFG['dataset_revision']}/{split}"
        for file in json.load(urllib.request.urlopen(api)):
            path = UPSTREAM / "data" / file["path"]
            hashes[file["path"]] = verify(path, file)
    (ROOT / "data/upstream_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    model, rev = CFG["model"], CFG["model_revision"]
    MODEL.mkdir(parents=True, exist_ok=True)
    files = json.load(
        urllib.request.urlopen(f"https://huggingface.co/api/models/{model}/tree/{rev}")
    )

    def obtain(file):
        name = file["path"]
        if name in [
            ".gitattributes",
            "README.md",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
        ]:
            return
        path = MODEL / name
        if not path.exists():
            transfer(
                path,
                f"https://huggingface.co/{model}/resolve/{rev}/{name}",
                file["size"],
            )
        digest = verify(path, file)
        print(f"Verified {name}", flush=True)
        return name, digest

    with ThreadPoolExecutor(4) as workers:
        model_hashes = dict(entry for entry in workers.map(obtain, files) if entry)
    (ROOT / "data/model_hashes.json").write_text(
        json.dumps(model_hashes, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
