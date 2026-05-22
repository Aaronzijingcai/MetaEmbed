# put all distributed training related code in this file
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process, supporting backward propagation.
    https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def dist_gather(x: torch.tensor):
    if not dist.is_initialized():
        return x
    if len(x.shape) == 0:
        x = x.reshape(1)
    x_gather = GatherLayer.apply(x)
    x_gather = torch.cat(x_gather, dim=0)
    return x_gather


class GatherLayerLavis(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad_lavis(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayerLavis.apply(tensors)
    return torch.cat(tensor_all, dim=0)


def gather_with_grad_torch(x: torch.Tensor) -> torch.Tensor:
    tensor_all = all_gather(x)
    return torch.cat(tensor_all, dim=0)


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() in [-1, 0]:
            print(*args)
    else:
        print("DIST NOT INITED: ", *args)


# applies to query embeddings of [B, D], padded to maximum B
# after all_gather, the global features will be restored to [B * (N-1) + remainder, D]
# **no gradient**
def all_gather_with_padding(local_image_features, world_size, debug=False):
    # local_image_features = torch.cat(local_image_features, dim=0).cuda()
    if world_size == 1:
        return local_image_features, [
            torch.tensor(
                [local_image_features.size(0)],
                dtype=torch.int64,
                device=local_image_features.device,
            )
        ]

    local_size = local_image_features.size(0)

    local_sizes = [torch.zeros(1, dtype=torch.int64).cuda() for _ in range(world_size)]
    dist.all_gather(local_sizes, torch.tensor([local_size], dtype=torch.int64).cuda())

    max_size = max([size.item() for size in local_sizes])

    if local_size < max_size:  # padding needed for this rank
        # print(f"Rank {dist.get_rank()} needs padding: {local_size} -> {max_size}")
        padding = torch.zeros(
            (max_size - local_size, *local_image_features.shape[1:]),
            dtype=local_image_features.dtype,
            device=local_image_features.device,
        )
        local_image_features = torch.cat([local_image_features, padding], dim=0)

    global_image_features = [
        torch.zeros_like(local_image_features) for _ in range(world_size)
    ]
    dist.all_gather(global_image_features, local_image_features)

    global_image_features = [
        feat[: size.item()] for feat, size in zip(global_image_features, local_sizes)
    ]
    if debug:
        print(f"Rank {dist.get_rank()} has local_sizes: {local_sizes}")
    global_image_features = torch.cat(global_image_features, dim=0)
    return global_image_features, local_sizes


def all_gather_with_padding_select_dim(
    local_tensor: torch.Tensor,
    world_size: int,
    pad_dim: int = 0,
    debug: bool = False,
):
    """
    All-gather a tensor across ranks with dynamic padding along a chosen dimension.

    Args:
        local_tensor (torch.Tensor): Local input tensor.
        world_size (int): Number of distributed ranks.
        pad_dim (int): Dimension along which padding is applied (default=0).
        debug (bool): Print debug info.

    Returns:
        global_tensor (torch.Tensor): Concatenated global tensor (unpadded).
        local_sizes (List[torch.Tensor]): Sizes before padding (per rank).
    """
    if world_size == 1:
        return local_tensor, [
            torch.tensor(
                [local_tensor.size(pad_dim)],
                dtype=torch.int64,
                device=local_tensor.device,
            )
        ]

    # Get local size along the chosen dimension
    local_size = local_tensor.size(pad_dim)

    # Gather all sizes
    local_sizes = [
        torch.zeros(1, dtype=torch.int64, device=local_tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(
        local_sizes,
        torch.tensor([local_size], dtype=torch.int64, device=local_tensor.device),
    )

    max_size = max(size.item() for size in local_sizes)

    # If smaller, pad along the chosen dimension
    if local_size < max_size:
        pad_shape = list(local_tensor.shape)
        pad_shape[pad_dim] = max_size - local_size
        padding_tensor = torch.zeros(
            pad_shape, dtype=local_tensor.dtype, device=local_tensor.device
        )
        local_tensor = torch.cat([local_tensor, padding_tensor], dim=pad_dim)

    # Allocate gather list
    global_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(global_tensors, local_tensor)

    if debug:
        print(
            f"Rank {dist.get_rank()} local_sizes={ [s.item() for s in local_sizes] }, pad_dim={pad_dim}"
        )

    # although padding happens at pad_dim, we still concat along dim=0
    return torch.cat(global_tensors, dim=0), local_sizes


def pad_to_max_len_right(x: torch.Tensor, world_size, debug=False):
    """
    Right-pad x along dim=1 so that all ranks share the same length.
    Args:
        x: Tensor of shape [B, L, D] (or [B, L] if 2D)
    Returns:
        Padded tensor of shape [B, max_L, D], with zeros on the right.
    """
    # 1) local length
    local_len = x.size(1)
    # 2) get global max length

    local_lens = [torch.zeros(1, dtype=torch.int64).cuda() for _ in range(world_size)]
    dist.all_gather(local_lens, torch.tensor([local_len], dtype=torch.int64).cuda())

    max_len = max([size.item() for size in local_lens])

    # 3) if shorter, pad on the right of dim=1
    if local_len < max_len:
        pad_amount = max_len - local_len
        # torch.nn.functional.pad takes (D_left, D_right, L_left, L_right)
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_amount), value=0.0)

    if debug:
        print(f"Rank {dist.get_rank()} has local_lens: {local_lens}")

    return x, local_lens


# simple all_gather function to gather embeddings across ranks
# identical shape garanteed by OrderedDistributedSampler and padded_to_max_len_right
def all_gather_without_grad(local_image_features, world_size):
    if world_size == 1:
        return local_image_features

    global_image_features = [
        torch.zeros_like(local_image_features) for _ in range(world_size)
    ]
    dist.all_gather(global_image_features, local_image_features)
    return torch.cat(global_image_features, dim=0)
