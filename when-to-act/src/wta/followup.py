"""Prepare the outcome-independent full judge set and reuse byte-identical frozen evaluations."""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict

from rapidfuzz import fuzz, process

from .common import (
    CFG,
    DATA,
    LABELS,
    MODEL,
    RAW,
    RESULTS,
    ROOT,
    call_error,
    parse_tools,
    read_rows,
    render,
    write_json,
    write_rows,
)
from .fetch import sha
from .prepare import key

DESIGN = DATA / "followup"
EXP = RESULTS / "expanded-generation"
MATCH = RESULTS / "matched-protocols"
CTRL = RESULTS / "direct-answer-controls"
DEST = {"generate": EXP, "mcq": MATCH, "control": CTRL}
MODELS = ["baseline"] + [
    f"non{p}-seed{s}" for p in (0, 10, 25, 50) for s in CFG["seeds"]
]
CORE = ["common.py", "evaluate.py", "decisions.py", "model.py"]


def digest(data):
    return hashlib.sha256(data.encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def clusters(raw):
    # Same union and iteration order as evaluate.final_rows, applied to the entire judge split.
    questions = defaultdict(set)
    for r in raw:
        questions[r["source_id"]].add(key(r.get("orig_question") or r["question"]))
    parents = {s: s for s in questions}

    def root(s):
        while parents[s] != s:
            parents[s] = parents[parents[s]]
            s = parents[s]
        return s

    owners = {}
    for s, qs in sorted(questions.items()):
        for q in sorted(qs):
            for near, _, _ in process.extract(
                q, list(owners), scorer=fuzz.ratio, score_cutoff=92, limit=None
            ):
                parents[root(s)] = root(owners[near])
            owners[q] = s
    return {s: root(s) for s in parents}


def normalized(r, group):
    choices = {
        k: [json.loads(v)] if k == "tool_call" else v for k, v in r["answers"].items()
    }
    assert set(choices) == set(LABELS)
    return dict(
        id=r["uuid"],
        group=group,
        source_id=r["source_id"],
        question=r["question"],
        tools=parse_tools(r["tools"]),
        label=r["correct_answer"],
        choices=choices,
        answer=choices[r["correct_answer"]],
    )


def fingerprints(row):
    prompt = render(row["question"], row["tools"])
    choices = [
        (k, render(row["question"], row["tools"], v).removesuffix("<|im_end|>\n"))
        for k, v in row["choices"].items()
    ]
    return digest(prompt), digest(
        json.dumps([prompt, choices], ensure_ascii=False, separators=(",", ":"))
    )


def prepare():
    from .freeze import main as check_freeze
    from .model import metadata_for

    check_freeze()
    base_hashes = read(DATA / "model_hashes.json")
    assert all(sha(MODEL / f) == h for f, h in base_hashes.items())
    data_hashes = read(DATA / "upstream_hashes.json")
    assert all(sha(RAW / f) == h for f, h in data_hashes.items())
    verified = {}
    current = metadata_for("baseline")
    for name in MODELS:
        hu = None
        if name != "baseline":
            folder = ROOT / "checkpoints" / name
            hu = sha(folder / "adapter_model.safetensors")
            c = read(folder / "adapter_config.json")
            assert (
                c["r"],
                c["lora_alpha"],
                set(c["target_modules"]),
                c["lora_dropout"],
            ) == (
                CFG["rank"],
                CFG["alpha"],
                set(CFG["lora_targets"]),
                CFG["lora_dropout"],
            )
            assert hu == read(RESULTS / name / "training.json")["adapter_sha256"]
        verified[name] = {
            "weight_sha256": hu,
            "adapter_config_sha256": None
            if hu is None
            else sha(folder / "adapter_config.json"),
        }
        for protocol in ("generate", "mcq"):
            old = read(RESULTS / name / f"official-{protocol}-metadata.json")
            for field in (
                "config",
                "packages",
                "gpu",
                "cuda_runtime",
                "template_sha256",
                "tokenizer_sha256",
                "lock_sha256",
                "dataset_hashes",
            ):
                assert current[field] == old[field], (name, field)
            assert all(
                old["implementation"][f] == current["implementation"][f] for f in CORE
            )
            assert old["weight_precision"] == "nf4" and old.get("adapter_sha256") == hu

    j, m = [
        read_rows(RAW / "test" / f"when2call_test_{s}.jsonl")
        for s in ("llm_judge", "mcq")
    ]
    jm, mm = [{r["uuid"]: r for r in rs} for rs in (j, m)]
    assert len(jm) == len(j) and len(mm) == len(m)
    groups = clusters(j)
    prepared = []
    for raw in sorted(j, key=lambda r: r["uuid"]):
        row = normalized(raw, groups[raw["source_id"]])
        match = mm.get(row["id"])
        # Require every released field and identical candidate order, fail visibly on disagreement.
        assert raw == match and list(raw["answers"]) == list(match["answers"]), row[
            "id"
        ]
        row["prompt_sha256"], row["candidates_sha256"] = fingerprints(row)
        assert fingerprints(normalized(match, row["group"])) == (
            row["prompt_sha256"],
            row["candidates_sha256"],
        )
        row["reference_status"] = (
            call_error(row["answer"], row["tools"]) or "valid"
            if row["label"] == "tool_call"
            else None
        )
        prepared.append(row)
    oldsets = {}
    cache = {}
    for protocol, lookup, index in (("generate", jm, 0), ("mcq", mm, 1)):
        ids = read_rows(DATA / f"official-{protocol}.jsonl")
        oldsets[protocol] = {r["id"] for r in ids}
        keys = defaultdict(list)
        for e in ids:
            r = normalized(lookup[e["id"]], e["group"])
            if protocol == "generate":
                assert r["group"] == groups[r["source_id"]]
            keys[fingerprints(r)[index]].append(r["id"])
        cache[protocol] = {}
        for r in prepared:
            eligible = keys[r[("prompt_sha256", "candidates_sha256")[index]]]
            if eligible:
                source = r["id"] if r["id"] in eligible else eligible[0]
                assert (
                    lookup[source]["correct_answer"] == r["label"]
                    and lookup[source]["answers"]["tool_call"]
                    == jm[r["id"]]["answers"]["tool_call"]
                )
                cache[protocol][r["id"]] = source
    assert oldsets["generate"] <= jm.keys()
    assert len({r["prompt_sha256"] for r in prepared}) == len(prepared)
    for r in prepared:
        r["original72"] = r["id"] in oldsets["generate"]
        r["reuse"] = {k: v[r["id"]] for k, v in cache.items() if r["id"] in v}
    status = Counter(r["reference_status"] for r in prepared if r["reference_status"])
    audit = dict(
        judge_rows=len(j),
        mcq_release_rows=len(m),
        full_record_matches=len(prepared),
        gold=Counter(r["label"] for r in prepared),
        distinct_rendered_prompts=len({r["prompt_sha256"] for r in prepared}),
        distinct_source_ids=len(groups),
        clusters=len(set(groups.values())),
        cluster_sizes=Counter(Counter(r["group"] for r in prepared).values()),
        previous_ids_in_judge={k: len(ids & jm.keys()) for k, ids in oldsets.items()},
        reusable_inputs_per_model={k: len(v) for k, v in cache.items()},
        reference_schema=status,
        invalid_references=[
            {"id": r["id"], "status": r["reference_status"], "call": r["answer"]}
            for r in prepared
            if r["reference_status"] not in (None, "valid")
        ],
        source_revisions=CFG,
        weight_and_adapter_config_hashes=verified,
        model_files=base_hashes,
        template_sha256=current["template_sha256"],
        tokenizer_sha256=current["tokenizer_sha256"],
        original_source_hashes={f: current["implementation"][f] for f in CORE},
        upstream=data_hashes,
        interpretation="Additional post-inspection analysis. Include every judge UUID, retain the original gold and report invalid references.",
    )
    write_rows(DESIGN / "official.jsonl", prepared)
    audit["manifest_sha256"] = sha(DESIGN / "official.jsonl")
    write_json(EXP / "audit.json", audit)
    print(
        {
            k: audit[k]
            for k in (
                "gold",
                "clusters",
                "cluster_sizes",
                "reusable_inputs_per_model",
                "reference_schema",
            )
        },
        flush=True,
    )


def inference(name, protocol):
    import torch
    from peft import PeftModel
    from .evaluate import generate, mcq
    from .model import load_model, metadata_for

    official = protocol != "control"
    manifest = DESIGN / ("official.jsonl" if official else "direct-families.jsonl")
    audit = read(EXP / "audit.json")
    rows = read_rows(manifest)
    signature = sha(manifest)
    if official:
        assert signature == audit["manifest_sha256"]
    else:
        assert signature == read(CTRL / "audit.json")["manifest_sha256"]
    folder = DEST[protocol] / "models"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"{name}.jsonl"
    meta_path = folder / f"{name}-metadata.json"
    old = read(RESULTS / name / "official-generate-metadata.json")
    meta = metadata_for(name)
    for f in CORE:
        assert old["implementation"][f] == meta["implementation"][f]
    for field in (
        "config",
        "packages",
        "cuda_runtime",
        "gpu",
        "template_sha256",
        "tokenizer_sha256",
        "lock_sha256",
    ):
        assert old[field] == meta[field]
    adapter = audit["weight_and_adapter_config_hashes"][name]
    if name != "baseline":
        ckpt = ROOT / "checkpoints" / name
        assert (
            sha(ckpt / "adapter_model.safetensors")
            == adapter["weight_sha256"]
            == old["adapter_sha256"]
        )
        assert sha(ckpt / "adapter_config.json") == adapter["adapter_config_sha256"]
    identity = dict(
        run=name,
        protocol=protocol,
        manifest_sha256=signature,
        adapter=adapter,
        backbone=CFG["model_revision"],
        weight_precision="nf4",
        template=meta["template_sha256"],
        lock=meta["lock_sha256"],
        core=audit["original_source_hashes"],
    )
    if output.exists():
        done = read(meta_path)
        assert done["identity"] == identity and done["prediction_sha256"] == sha(output)
        return
    partial = output.with_suffix(".incomplete")
    marker = output.with_suffix(".resume.json")
    if partial.exists():
        assert read(marker) == identity
    write_json(marker, identity)
    done = {p["id"]: p for p in read_rows(partial)} if partial.exists() else {}
    cache = (
        {r["id"]: r for r in read_rows(RESULTS / name / f"official-{protocol}.jsonl")}
        if official
        else {}
    )
    pending = []
    for r in rows:
        if r["id"] in done:
            continue
        source = r.get("reuse", {}).get(protocol)
        if source is None:
            pending.append(r)
        else:
            p = cache[source]
            assert p["gold"] == r["label"] and p["n_tools"] == len(r["tools"])
            done[r["id"]] = dict(p, id=r["id"], group=r["group"], reused_id=source)
    pending.sort(key=lambda r: len(render(r["question"], r["tools"])))
    batch = 1 if protocol == "mcq" else CFG["eval_batch_size"]
    if pending:
        model = load_model(nf4=True)
        if name != "baseline":
            model = PeftModel.from_pretrained(model, ckpt)
        model.eval()
        assert model.is_loaded_in_4bit
        with partial.open("a") as f:
            i = 0
            while i < len(pending):
                pack = pending[i : i + batch]
                try:
                    preds = (
                        [mcq(model, pack[0])]
                        if protocol == "mcq"
                        else generate(model, pack)
                    )
                except torch.OutOfMemoryError:
                    if batch == 1:
                        raise
                    batch = max(1, batch // 2)
                    torch.cuda.empty_cache()
                    continue
                for pred in preds:
                    pred["reused_id"] = None
                    done[pred["id"]] = pred
                    f.write(json.dumps(pred, ensure_ascii=False) + "\n")
                f.flush()
                i += len(pack)
                if i % 24 == 0 or i == len(pending):
                    print(name, protocol, i, len(pending), flush=True)
    assert len(done) == len(rows)
    tmp = output.with_suffix(".complete.tmp")
    write_rows(tmp, (done[r["id"]] for r in rows))
    meta.update(
        identity=identity,
        rows=len(rows),
        reused=sum(p["reused_id"] is not None for p in done.values()),
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
        prediction_sha256=sha(tmp),
    )
    write_json(meta_path, meta)
    os.replace(tmp, output)
    partial.unlink(missing_ok=True)
    marker.unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("protocol", choices=["prepare", *DEST])
    p.add_argument("name", nargs="?", choices=MODELS)
    a = p.parse_args()
    if a.protocol == "prepare":
        prepare()
    else:
        inference(a.name, a.protocol)
