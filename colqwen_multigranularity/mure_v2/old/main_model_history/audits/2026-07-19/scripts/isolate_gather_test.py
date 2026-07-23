import os

import torch
import torch.distributed as dist

from colpali_engine.utils.dist_utils import gather_with_grad_torch


def run(mode: str):
    os.environ["MURE_GATHER_WITH_GRAD_MODE"] = mode
    rank = dist.get_rank()
    x = (torch.arange(6, device="cuda", dtype=torch.float32) + rank * 10).reshape(2, 3)
    x.requires_grad_(True)
    gathered = gather_with_grad_torch(x)
    weights = torch.arange(1, gathered.numel() + 1, device="cuda", dtype=torch.float32).reshape_as(gathered)
    loss = (rank + 1) * (gathered * weights).sum()
    loss.backward()
    return gathered.detach(), x.grad.detach()


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    legacy_output, legacy_grad = run("torch_default_group")
    dist.barrier()
    isolated_output, isolated_grad = run("torch")

    output_diff = (legacy_output - isolated_output).abs().max()
    grad_diff = (legacy_grad - isolated_grad).abs().max()
    expected_grad = 36 * torch.arange(
        dist.get_rank() * 6 + 1,
        dist.get_rank() * 6 + 7,
        device="cuda",
        dtype=torch.float32,
    ).reshape(2, 3)
    expected_diff = (isolated_grad - expected_grad).abs().max()
    metrics = torch.stack([output_diff, grad_diff, expected_diff])
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        print(
            f"output_max_diff={metrics[0].item()} "
            f"gradient_max_diff={metrics[1].item()} "
            f"expected_gradient_max_diff={metrics[2].item()}",
            flush=True,
        )
    if metrics.max().item() != 0.0:
        raise SystemExit(1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
