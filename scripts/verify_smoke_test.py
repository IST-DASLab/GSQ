"""Distributed smoke test for GSQ in a multi-node Slurm job.

Launched by verify_setup.sbatch.sh via torchrun. Each rank:
  1. Prints identity (rank, node, GPU)
  2. NCCL all-reduce correctness
  3. bf16 matmul
  4. flash-attn (optional)
  5. All-gather across all ranks
  6. HuggingFace transformers import
  7. GSQ src.trainer import
"""

import os
import socket
import sys

import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    failures = []

    # ── 1. Identity ───────────────────────────────────────────────────────
    gpu_name = torch.cuda.get_device_name(local_rank)
    total_mem_gb = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
    node = socket.gethostname()
    print(
        f"[rank {rank:02d}/{world_size-1}] node={node}  local_rank={local_rank}"
        f"  gpu={gpu_name}  mem={total_mem_gb:.0f} GB",
        flush=True,
    )

    # ── 2. NCCL all-reduce correctness ────────────────────────────────────
    t = torch.tensor(float(rank), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = float(world_size * (world_size - 1) / 2)
    if abs(t.item() - expected) > 1e-3:
        failures.append(f"all-reduce: got {t.item()}, expected {expected}")
    elif rank == 0:
        print(f"[rank 00] NCCL all-reduce OK  (sum={t.item():.0f})", flush=True)

    # ── 3. bf16 matmul ────────────────────────────────────────────────────
    try:
        a = torch.randn(512, 512, dtype=torch.bfloat16, device=device)
        b = torch.randn(512, 512, dtype=torch.bfloat16, device=device)
        c = torch.matmul(a, b)
        assert c.shape == (512, 512)
        if rank == 0:
            print("[rank 00] bf16 matmul OK", flush=True)
    except Exception as e:
        failures.append(f"bf16 matmul: {e}")

    # ── 4. flash-attn (optional) ──────────────────────────────────────────
    try:
        from flash_attn import flash_attn_func

        B, S, H, D = 1, 128, 8, 64
        q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
        out = flash_attn_func(q, k, v)
        assert out.shape == (B, S, H, D)
        if rank == 0:
            import flash_attn as fa

            print(
                f"[rank 00] flash-attn OK  (version {fa.__version__})", flush=True
            )
    except ImportError:
        if rank == 0:
            print("[rank 00] flash-attn not installed (optional)", flush=True)
    except Exception as e:
        failures.append(f"flash-attn: {e}")

    # ── 5. All-gather across all ranks ────────────────────────────────────
    local_t = torch.tensor([rank], dtype=torch.int64, device=device)
    gathered = [
        torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)
    ]
    dist.all_gather(gathered, local_t)
    collected = [x.item() for x in gathered]
    expected_list = list(range(world_size))
    if collected != expected_list:
        failures.append(f"all-gather: got {collected}, expected {expected_list}")
    elif rank == 0:
        print(f"[rank 00] all-gather OK  (ranks={collected})", flush=True)

    # ── 6. HuggingFace import ─────────────────────────────────────────────
    try:
        import transformers

        if rank == 0:
            print(
                f"[rank 00] transformers OK  ({transformers.__version__})", flush=True
            )
    except ImportError as e:
        failures.append(f"transformers import: {e}")

    # ── 7. GSQ src import ─────────────────────────────────────────────────
    try:
        from src.trainer import QuantizationTrainer  # noqa: F401

        if rank == 0:
            print("[rank 00] src.trainer OK", flush=True)
    except Exception as e:
        failures.append(f"src.trainer import: {e}")

    # ── 8. Barrier ────────────────────────────────────────────────────────
    dist.barrier()

    # ── Summary ───────────────────────────────────────────────────────────
    if failures:
        print(f"[rank {rank:02d}] FAILED: {failures}", flush=True)
        dist.destroy_process_group()
        sys.exit(1)
    else:
        if rank == 0:
            print(flush=True)
            print(f"ALL CHECKS PASSED on rank {rank}/{world_size-1}", flush=True)
            print(f"  PyTorch  : {torch.__version__}", flush=True)
            print(f"  CUDA     : {torch.version.cuda}", flush=True)
            print(
                f"  World    : {world_size} ranks across"
                f' {os.environ.get("SLURM_NNODES", "?")} nodes',
                flush=True,
            )
            print(flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
