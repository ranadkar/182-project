"""Exercise NF4 CUDA backprop and check the actual training objective numerically."""

from types import SimpleNamespace

import bitsandbytes as bnb
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

from .common import RESULTS, render, selected, tokens, write_json
from .evaluate import mcq
from .model import load_model, metadata_for
from .train import loss_on


@torch.inference_mode()
def verify_mcq():
    # Training-side fixture only. These candidate texts have no benchmark score.
    row = next(r for r in selected("dev") if r["label"] == "tool_call")
    row["choices"] = {
        "direct": "The answer is ten.",
        "tool_call": row["answer"],
        "request_for_info": "Could you specify which entry you want?",
        "cannot_answer": "I am unable to retrieve that information.",
    }
    model = load_model().eval()
    batched = mcq(model, row)
    prefix = tokens(render(row["question"], row["tools"]))
    errors = {}
    for label, answer in row["choices"].items():
        full = render(row["question"], row["tools"], answer).removesuffix(
            "<|im_end|>\n"
        )
        ids = tokens(full)
        assert ids[: len(prefix)] == prefix
        x = torch.tensor(ids, device="cuda")[None, :]
        # Serial scoring with full output head, no right padding, explicit gather.
        logits = (
            model(input_ids=x, use_cache=False).logits[0, len(prefix) - 1 : -1].float()
        )
        target = x[0, len(prefix) :]
        score = logits.log_softmax(-1).gather(1, target[:, None]).sum().item()
        errors[label] = abs(score - batched["log_likelihood"][label]) / len(target)
    if max(errors.values()) > 0.05:
        raise ValueError(f"Batched vs serial MCQ likelihood disagreement: {errors}")
    return {"fixture_id": row["id"], "absolute_error_nats_per_token": errors}


def main():
    layer = bnb.nn.Linear4bit(
        128, 64, quant_type="nf4", compute_dtype=torch.bfloat16
    ).to("cuda")
    x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    layer(x).float().square().mean().backward()
    assert torch.isfinite(x.grad).all()

    class TinyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(13, 6)
            self.head = nn.Linear(6, 13)

        def forward(self, input_ids, use_cache=False, logits_to_keep=None):
            z = self.head(self.embedding(input_ids))
            return SimpleNamespace(
                logits=z[:, -logits_to_keep:] if logits_to_keep else z
            )

    model = TinyHead().cuda()
    ids = [1, 5, 9, 6, 5, 7, 3]
    sliced = loss_on(model, (ids, 4))
    logits = model(torch.tensor([ids[:-1]], device="cuda")).logits
    reference = F.cross_entropy(
        logits.squeeze(0), torch.tensor([-100, -100, -100, 5, 7, 3], device="cuda")
    )
    torch.testing.assert_close(sliced, reference)
    for a, b in zip(
        torch.autograd.grad(sliced, list(model.parameters())),
        torch.autograd.grad(reference, list(model.parameters())),
    ):
        torch.testing.assert_close(a, b)
    metadata = metadata_for("hardware")
    kernel = qwen.chunk_gated_delta_rule
    metadata.update(
        nf4_forward_backward=True,
        masked_gradient_equivalence=True,
        delta_kernel=kernel.__module__ + "." + kernel.__name__
        if kernel
        else "torch fallback",
        causal_conv1d=qwen.causal_conv1d_fn is not None,
        free_total_mib=[v / 2**20 for v in torch.cuda.mem_get_info()],
    )
    metadata["mcq_serial_comparison"] = verify_mcq()
    write_json(RESULTS / "hardware.json", metadata)
    print(
        metadata["gpu"],
        metadata["capability"],
        metadata["cuda_runtime"],
        metadata["delta_kernel"],
    )


if __name__ == "__main__":
    main()
