from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class MLPShardSpec:
    tensor_name: str
    shard_dim: int
    start: int
    end: int
    global_shape: tuple[int, int]

    @property
    def local_size(self) -> int:
        return self.end - self.start


class _RowParallelSum(torch.autograd.Function):
    """SUM partial row-parallel outputs; backward is identity.

    Each tensor-parallel rank computes the same scalar loss from the same
    replicated batch. For y = sum_r y_r, each local branch needs the same
    dL/dy. A normal autograd-aware all-reduce would all-reduce the backward
    gradient again and scale it by world_size.
    """

    @staticmethod
    def forward(ctx, partial: torch.Tensor, group):
        out = partial.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        ctx.group = group
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def row_parallel_sum(partial: torch.Tensor, group=None) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size(group=group) == 1:
        return partial
    return _RowParallelSum.apply(partial, group)


class DenseMLPTensorParallel:
    """Tensor-parallel utilities for dense gated MLP GSQ training.

    Sharding convention:
      gate_proj: shard weight/scales on dim 0
      up_proj:   shard weight/scales on dim 0
      down_proj: shard weight on dim 1 and scales on group-column dim 1
    """

    SUPPORTED_LEAVES = {"gate_proj", "up_proj", "down_proj"}

    def __init__(
        self,
        rank: int,
        world_size: int,
        groupsize: int,
        group=None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.groupsize = groupsize
        self.group = group
        self.specs: dict[str, MLPShardSpec] = {}

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    def validate_tensor_name(self, tensor_name: str) -> str:
        leaf = tensor_name.rsplit(".", 1)[-1]
        if leaf not in self.SUPPORTED_LEAVES:
            raise ValueError(
                f"tensor_sharded mode only supports dense gated-MLP tensors; "
                f"got {tensor_name!r}"
            )
        return leaf

    def _equal_range(self, size: int) -> tuple[int, int]:
        if size % self.world_size != 0:
            raise ValueError(
                f"Sharded dimension {size} must be divisible by "
                f"world_size={self.world_size} in the first implementation."
            )
        chunk = size // self.world_size
        start = self.rank * chunk
        return start, start + chunk

    def shard_quantizer_inputs(
        self,
        tensor_name: str,
        Q: torch.Tensor,
        scales: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, MLPShardSpec]:
        leaf = self.validate_tensor_name(tensor_name)

        if Q.ndim != 2 or scales.ndim != 2:
            raise ValueError(
                f"Expected 2D Q/scales for {tensor_name}, got "
                f"Q={tuple(Q.shape)}, scales={tuple(scales.shape)}"
            )

        global_shape = tuple(Q.shape)

        if leaf in {"gate_proj", "up_proj"}:
            start, end = self._equal_range(Q.shape[0])
            q_local = Q[start:end, :].contiguous()
            scales_local = scales[start:end, :].contiguous()
            shard_dim = 0
        else:
            start, end = self._equal_range(Q.shape[1])
            if start % self.groupsize != 0 or end % self.groupsize != 0:
                raise ValueError(
                    f"{tensor_name}: down_proj shard [{start}:{end}] is not "
                    f"aligned to groupsize={self.groupsize}"
                )
            group_start = start // self.groupsize
            group_end = end // self.groupsize
            q_local = Q[:, start:end].contiguous()
            scales_local = scales[:, group_start:group_end].contiguous()
            shard_dim = 1

        spec = MLPShardSpec(
            tensor_name=tensor_name,
            shard_dim=shard_dim,
            start=start,
            end=end,
            global_shape=global_shape,
        )
        self.specs[tensor_name] = spec
        return q_local, scales_local, spec

    def spec(self, tensor_name: str) -> MLPShardSpec:
        try:
            return self.specs[tensor_name]
        except KeyError as exc:
            raise KeyError(f"No shard spec registered for {tensor_name!r}") from exc

    def gather_equal_gpu_shards_to_rank0_cpu(
        self,
        tensor_name: str,
        local: torch.Tensor,
        *,
        scale_tensor: bool = False,
    ) -> torch.Tensor | None:
        """Reassemble equal-sized shards on rank 0 without a full GPU gather.

        Rank 0 receives one peer shard at a time into a local GPU buffer and
        immediately copies it to CPU. This avoids materializing the full tensor
        on rank 0's GPU.
        """
        if self.world_size == 1:
            return local.detach().cpu()

        spec = self.spec(tensor_name)
        cat_dim = spec.shard_dim
        if scale_tensor and spec.shard_dim == 1:
            cat_dim = 1

        local = local.detach().contiguous()

        if self.rank == 0:
            pieces = [local.cpu()]
            recv = torch.empty_like(local)
            for src in range(1, self.world_size):
                dist.recv(recv, src=src, group=self.group)
                pieces.append(recv.cpu())
            return torch.cat(pieces, dim=cat_dim)

        dist.send(local, dst=0, group=self.group)
        return None
