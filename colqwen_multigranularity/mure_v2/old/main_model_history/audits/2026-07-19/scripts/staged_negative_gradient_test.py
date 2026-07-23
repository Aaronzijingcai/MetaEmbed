import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


PROJECT = "/MURE-V2/code/MetaEmbed/colqwen_multigranularity"
sys.path.insert(0, f"{PROJECT}/vendor")

from colpali_engine.utils.dist_utils import (  # noqa: E402
    gather_with_grad_torch,
    sync_gradients_after_backward,
)


class SharedEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(5, 4, bias=True)

    def forward(self, inputs):
        return torch.tanh(self.proj(inputs))


def run(model, query_inputs, doc_inputs, neg_inputs, *, staged):
    model.zero_grad(set_to_none=True)
    with model.no_sync():
        query = model(query_inputs)
        doc = model(doc_inputs)
        negative = model(neg_inputs)
        gathered_doc = gather_with_grad_torch(doc)
        if staged:
            negative_leaf = negative.detach().requires_grad_(True)
            negative_for_loss = negative_leaf
        else:
            negative_leaf = None
            negative_for_loss = negative

        positive_scores = query @ gathered_doc.transpose(0, 1)
        negative_scores = (query * negative_for_loss).sum(dim=1, keepdim=True)
        logits = torch.cat((positive_scores, negative_scores), dim=1)
        labels = torch.arange(query.size(0), device=query.device)
        labels.add_(dist.get_rank() * query.size(0))
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        if staged:
            torch.autograd.backward(negative, negative_leaf.grad)

    sync_gradients_after_backward(model, bucket_bytes=1024)
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    os.environ["MURE_GATHER_WITH_GRAD_MODE"] = "torch"

    torch.manual_seed(20260719)
    model = DDP(
        SharedEncoder().to(device),
        device_ids=[local_rank],
        output_device=local_rank,
    )
    base = torch.arange(15, device=device, dtype=torch.float32).view(3, 5)
    query_inputs = base + rank * 0.1
    doc_inputs = base.flip(0) + rank * 0.2
    neg_inputs = base.roll(1, 0) - rank * 0.15

    full = run(model, query_inputs, doc_inputs, neg_inputs, staged=False)
    dist.barrier()
    staged = run(model, query_inputs, doc_inputs, neg_inputs, staged=True)

    max_diff = max(
        (full[name] - staged[name]).abs().max().item()
        for name in full
    )
    result = torch.tensor(max_diff, device=device)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"staged_negative_gradient_max_diff={result.item():.9g}", flush=True)
    if result.item() > 1e-6:
        raise AssertionError(f"staged gradient mismatch: {result.item()}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
