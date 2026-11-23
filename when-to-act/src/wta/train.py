"""Single-GPU, one-pass QLoRA with response-only causal cross-entropy."""

import argparse
import json
import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import set_seed

from .common import CFG, DATA, RESULTS, ROOT, encode, selected, write_json, write_rows
from .fetch import sha
from .model import load_model, metadata_for, run_name


def loss_on(model, row):
    ids, boundary = row
    x = torch.tensor(ids, device="cuda").unsqueeze(0)
    # Predict token i from i-1. Only assistant tokens, including the end-of-turn,
    # contribute. Slicing before the 248k-way head also limits peak VRAM.
    out = model(
        input_ids=x[:, :-1], use_cache=False, logits_to_keep=x.shape[1] - boundary
    )
    return torch.nn.functional.cross_entropy(out.logits[0].float(), x[0, boundary:])


@torch.no_grad()
def validation(model, examples):
    model.eval()
    value = sum(loss_on(model, row).item() for row in examples) / len(examples)
    model.train()
    return value


def train(percent, seed):
    name = run_name(percent, seed)
    folder = RESULTS / name
    folder.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model = load_model()
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=CFG["rank"],
            lora_alpha=CFG["alpha"],
            target_modules=CFG["lora_targets"],
            lora_dropout=CFG["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.train()
    rows = selected(f"train-{percent}")
    random.Random(seed).shuffle(rows)
    ordered = [encode(r) for r in rows]
    val = [encode(r) for r in selected("dev")]
    group = CFG["gradient_accumulation"]
    assert len(ordered) % group == 0 and all(
        len(x) <= CFG["max_train_tokens"] for x, _ in ordered
    )
    steps = len(rows) // group
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params, lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"], fused=True
    )
    warmup = CFG["warmup_steps"]
    schedule = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda t: (t + 1) / warmup
        if t < warmup
        else 0.5 * (1 + math.cos(math.pi * (t - warmup + 1) / (steps - warmup + 1))),
    )
    meta = metadata_for(name)
    meta.update(
        manifest_sha256=sha(DATA / f"train-{percent}.jsonl"),
        examples=len(rows),
        steps=steps,
        seed=seed,
        percent=percent,
        trainable_parameters=sum(p.numel() for p in params),
        training_tokens=sum(len(x) for x, _ in ordered),
        assistant_tokens=sum(len(x) - b for x, b in ordered),
        training_order=[r["id"] for r in rows],
    )
    write_json(folder / "training.json", meta)
    start = time.monotonic()
    write_rows(
        folder / "loss.jsonl", [{"step": 0, "validation_loss": validation(model, val)}]
    )
    for step in range(1, steps + 1):
        total = 0.0
        for row in ordered[(step - 1) * group : step * group]:
            loss = loss_on(model, row)
            (loss / group).backward()
            total += loss.detach().item() / group
        norm = torch.nn.utils.clip_grad_norm_(
            params, CFG["max_grad_norm"], error_if_nonfinite=True
        ).item()
        lr = opt.param_groups[0]["lr"]
        opt.step()
        schedule.step()
        opt.zero_grad(set_to_none=True)
        item = {
            "step": step,
            "loss": total,
            "gradient_norm": norm,
            "lr": lr,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "elapsed_s": time.monotonic() - start,
        }
        if step in {steps // 4, steps // 2, 3 * steps // 4, steps}:
            item["validation_loss"] = validation(model, val)
        with (folder / "loss.jsonl").open("a") as f:
            f.write(json.dumps(item) + "\n")
        if step == 1 or step % 10 == 0:
            print(name, item, flush=True)
    ckpt = ROOT / "checkpoints" / name
    model.save_pretrained(ckpt)
    meta["adapter_sha256"] = sha(ckpt / "adapter_model.safetensors")
    meta["elapsed_s"] = time.monotonic() - start
    write_json(folder / "training.json", meta)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("percent", type=int, choices=CFG["non_tool_percentages"])
    p.add_argument("seed", type=int, choices=CFG["seeds"])
    args = p.parse_args()
    train(args.percent, args.seed)
