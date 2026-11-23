"""Greedy free generation and four-continuation likelihood evaluation."""

import argparse
import json
import os
import random
from collections import Counter, defaultdict

import torch
from peft import PeftModel
from rapidfuzz import fuzz, process

from .common import (
    CFG,
    DATA,
    RAW,
    RESULTS,
    ROOT,
    call_error,
    parse_tools,
    read_rows,
    render,
    selected,
    tokenizer,
    tokens,
    write_json,
    write_rows,
)
from .decisions import judge
from .fetch import sha
from .model import load_model, metadata_for
from .prepare import key


def final_rows(protocol):
    filename = (
        "when2call_test_mcq.jsonl"
        if protocol == "mcq"
        else "when2call_test_llm_judge.jsonl"
    )
    raw = read_rows(RAW / "test" / filename)
    # Resample related BFCL variants together, including near-duplicate source questions.
    source_questions = defaultdict(set)
    for r in raw:
        source_questions[r["source_id"]].add(
            key(r.get("orig_question") or r["question"])
        )
    parents = {s: s for s in source_questions}

    def root(s):
        while parents[s] != s:
            parents[s] = parents[parents[s]]
            s = parents[s]
        return s

    owners = {}
    for s, questions in sorted(source_questions.items()):
        for q in sorted(questions):
            for neighbor, _, _ in process.extract(
                q, list(owners), scorer=fuzz.ratio, score_cutoff=92, limit=None
            ):
                parents[root(s)] = root(owners[neighbor])
            owners[q] = s
    groups = sorted(set(root(s) for s in parents))
    rng = random.Random(CFG["split_seed"])
    rng.shuffle(groups)
    allowed = (
        set(groups[: CFG["official_mcq_groups"]]) if protocol == "mcq" else set(groups)
    )
    rows = []
    repeated = defaultdict(list)
    for r in sorted(raw, key=lambda r: r["uuid"]):
        group = root(r["source_id"])
        if group not in allowed:
            continue
        tools = parse_tools(r["tools"])
        menu = json.dumps(tools, sort_keys=True)
        # Only discard literal duplicate prompt content. Different menus are distinct decisions.
        fingerprint = (r["question"].strip(), menu)
        repeated[fingerprint].append((r, tools, group))
    conflict_rows = 0
    for duplicates in repeated.values():
        if len({r["correct_answer"] for r, _, _ in duplicates}) > 1:
            conflict_rows += len(duplicates)
            continue
        r, tools, group = duplicates[0]
        call = json.loads(r["answers"]["tool_call"])
        choices = {
            k: ([call] if k == "tool_call" else v) for k, v in r["answers"].items()
        }
        rows.append(
            {
                "id": r["uuid"],
                "group": group,
                "question": r["question"],
                "tools": tools,
                "label": r["correct_answer"],
                "choices": choices,
                "answer": choices[r["correct_answer"]],
            }
        )
    if protocol == "generate":
        by_label = defaultdict(list)
        for row in rows:
            by_label[row["label"]].append(row)
        sample = random.Random(CFG["split_seed"])
        rows = [
            r
            for label in sorted(by_label)
            for r in sample.sample(
                sorted(by_label[label], key=lambda r: r["id"]),
                CFG["official_generation_per_class"],
            )
        ]
    write_json(
        RESULTS / f"official-{protocol}-data-audit.json",
        {
            "source_file": filename,
            "selected_rows": len(rows),
            "selected_groups": len({r["group"] for r in rows}),
            "gold": Counter(r["label"] for r in rows),
            "conflicting_prompt_rows_excluded": conflict_rows,
            "repeated_prompt_copies_before_sampling": sum(
                len(v) - 1 for v in repeated.values()
            ),
            "required_reference_schema_status": Counter(
                call_error(r["answer"], r["tools"]) or "valid"
                for r in rows
                if r["label"] == "tool_call"
            ),
        },
    )
    write_rows(
        DATA / f"official-{protocol}.jsonl",
        ({"id": r["id"], "group": r["group"]} for r in rows),
    )
    return rows


def base_record(row):
    return {
        "id": row["id"],
        "group": row.get("group", row["id"]),
        "gold": row["label"],
        "n_tools": len(row["tools"]),
    }


@torch.inference_mode()
def generate(model, rows):
    tok = tokenizer()
    tok.padding_side = "left"
    enc = tok(
        [render(r["question"], r["tools"]) for r in rows],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    out = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=CFG["max_new_tokens"],
        use_cache=True,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
    )
    results = []
    for row, seq in zip(rows, out[:, enc["input_ids"].shape[1] :].tolist()):
        eos = tok.convert_tokens_to_ids("<|im_end|>")
        stopped = eos in seq
        text = tok.decode(
            seq[: seq.index(eos)] if stopped else seq, skip_special_tokens=False
        )
        label, error, calls = judge(text, row["tools"])
        results.append(
            dict(
                base_record(row),
                prediction=label,
                output=text,
                call_error=error,
                length_capped=not stopped,
                calls=calls,
                exact_call=(calls == row["answer"] and error is None)
                if row["label"] == "tool_call"
                else None,
            )
        )
    return results


@torch.inference_mode()
def mcq(model, row):
    prompt = render(row["question"], row["tools"])
    prefix = tokens(prompt)
    sums = {}
    norm = {}
    candidates = []
    for label, answer in row["choices"].items():
        full = render(row["question"], row["tools"], answer).removesuffix(
            "<|im_end|>\n"
        )
        assert full.startswith(prompt)
        suffix = full[len(prompt) :]
        ids = tokens(full)
        if ids[: len(prefix)] != prefix:
            # Inference starts at the already-tokenized generation prompt.
            ids = prefix + tokens(suffix)
        candidates.append((label, ids, len(suffix.encode("utf-8"))))

    def score_batch(batch):
        width = max(len(ids) for _, ids, _ in batch)
        batch_ids = torch.full(
            (len(batch), width),
            tokenizer().pad_token_id,
            device="cuda",
            dtype=torch.long,
        )
        mask = torch.zeros_like(batch_ids)
        for i, (_, ids, _) in enumerate(batch):
            batch_ids[i, : len(ids)] = torch.tensor(ids, device="cuda")
            mask[i, : len(ids)] = 1
        logits = model(
            input_ids=batch_ids[:, :-1],
            attention_mask=mask[:, :-1],
            use_cache=False,
            logits_to_keep=width - len(prefix),
        ).logits
        for i, (label, ids, nbytes) in enumerate(batch):
            n = len(ids) - len(prefix)
            score = -torch.nn.functional.cross_entropy(
                logits[i, :n].float(),
                batch_ids[i, len(prefix) : len(ids)],
                reduction="sum",
            ).item()
            sums[label] = score
            norm[label] = score / nbytes

    # Long tool menus can exceed laptop memory when scored four at once.
    try:
        score_batch(candidates)
    except torch.OutOfMemoryError:
        pass
    if len(sums) != len(candidates):
        torch.cuda.empty_cache()
        for candidate in candidates:
            score_batch([candidate])
    return dict(
        base_record(row),
        prediction=max(norm, key=norm.get),
        prediction_raw=max(sums, key=sums.get),
        log_likelihood=sums,
        log_likelihood_per_byte=norm,
    )


def evaluate(name, split, protocol):
    if split == "official" and not (RESULTS / "freeze.json").exists():
        raise RuntimeError(
            "Freeze the completed pilot configuration before evaluating official prompts"
        )
    if split == "official":
        from .freeze import main as verify_freeze

        verify_freeze()
    rows = selected("dev") if split == "dev" else final_rows(protocol)
    if protocol == "mcq" and split == "dev":
        raise ValueError("Preference dev has no four-way candidate set")
    folder = RESULTS / name
    folder.mkdir(exist_ok=True, parents=True)
    path = folder / f"{split}-{protocol}.jsonl"
    if name == "bf16-reference" and split != "dev":
        raise ValueError("The bf16 precision check is development-only")
    model = load_model(nf4=name != "bf16-reference")
    meta = metadata_for(name)
    meta["weight_precision"] = "bf16" if name == "bf16-reference" else "nf4"
    if name not in {"baseline", "bf16-reference"}:
        ckpt = ROOT / "checkpoints" / name
        model = PeftModel.from_pretrained(model, ckpt)
        meta["adapter_sha256"] = sha(ckpt / "adapter_model.safetensors")
    model.eval()
    rows.sort(key=lambda r: len(render(r["question"], r["tools"])))
    batch = CFG["eval_batch_size"] if protocol == "generate" else 1
    temp = path.with_suffix(".incomplete")
    marker = temp.with_suffix(".resume.json")
    identity = {
        "split": split,
        "protocol": protocol,
        "model": name,
        "ids": [r["id"] for r in rows],
        "config": CFG,
        "lock": meta["lock_sha256"],
        "adapter": meta.get("adapter_sha256"),
        "sources": {
            k: meta["implementation"][k]
            for k in ("evaluate.py", "common.py", "model.py", "decisions.py")
        },
    }
    if temp.exists() and json.loads(marker.read_text()) != identity:
        raise RuntimeError(
            f"Partial predictions in {temp} use a different configuration"
        )
    write_json(marker, identity)
    done = read_rows(temp) if temp.exists() else []
    assert [r["id"] for r in rows[: len(done)]] == [r["id"] for r in done]
    i = len(done)
    with temp.open("a") as f:
        while i < len(rows):
            pack = rows[i : i + batch]
            try:
                preds = (
                    generate(model, pack)
                    if protocol == "generate"
                    else [mcq(model, pack[0])]
                )
            except torch.OutOfMemoryError:
                if batch == 1:
                    raise
                batch = max(1, batch // 2)
                torch.cuda.empty_cache()
                continue
            for pred in preds:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            f.flush()
            i += len(preds)
            if i % 16 == 0 or i == len(rows):
                print(name, split, protocol, i, "/", len(rows), flush=True)
    meta.update(
        split=split,
        protocol=protocol,
        examples=len(rows),
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
    )
    write_json(folder / f"{split}-{protocol}-metadata.json", meta)
    os.replace(temp, path)
    marker.unlink()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("split", choices=["dev", "official"])
    p.add_argument("protocol", choices=["generate", "mcq"])
    a = p.parse_args()
    evaluate(a.name, a.split, a.protocol)
