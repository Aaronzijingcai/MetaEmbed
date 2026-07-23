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


class BranchedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Linear(4, 3, bias=False)
        self.right = torch.nn.Linear(4, 3, bias=False)

    def forward(self, x, use_left):
        return self.left(x) if use_left else self.right(x)


def run_backward(model, x, coefficients, *, deferred):
    model.zero_grad(set_to_none=True)
    context = model.no_sync() if deferred else torch.enable_grad()
    os.environ["MURE_GATHER_WITH_GRAD_MODE"] = (
        "torch" if deferred else "torch_default_group"
    )
    with context:
        local_docs = model(x, dist.get_rank() % 2 == 0)
        global_docs = gather_with_grad_torch(local_docs)
        loss = (global_docs * coefficients).sum()
        loss.backward()
    if deferred:
        sync_gradients_after_backward(model, bucket_bytes=1024)
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(1234)
    model = BranchedModel().to(device)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )
    x = torch.arange(8, device=device, dtype=torch.float32).view(2, 4) + rank
    coefficients = (
        torch.arange(dist.get_world_size() * 6, device=device, dtype=torch.float32)
        .view(dist.get_world_size() * 2, 3)
        .add_(rank * 0.25)
    )

    baseline = run_backward(model, x, coefficients, deferred=False)
    dist.barrier()
    deferred = run_backward(model, x, coefficients, deferred=True)

    max_diff = 0.0
    for name in baseline:
        if baseline[name] is None or deferred[name] is None:
            if baseline[name] is not deferred[name]:
                raise AssertionError(f"gradient presence mismatch for {name}")
            continue
        max_diff = max(max_diff, (baseline[name] - deferred[name]).abs().max().item())

    result = torch.tensor(max_diff, device=device)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"deferred_ddp_gradient_max_diff={result.item():.9g}", flush=True)
    if result.item() > 1e-5:
        raise AssertionError(f"gradient mismatch: {result.item()}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
