"""Minimal NCCL all_to_all diagnostic for Clariden (GH200 + Slingshot 11)."""
import os
import sys
import torch
import torch.distributed as dist

def log(msg):
    rank = dist.get_rank()
    if rank == 0:
        print(msg, flush=True)

def test_all_reduce():
    rank = dist.get_rank()
    t = torch.tensor([float(rank)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(range(dist.get_world_size()))
    assert abs(t.item() - expected) < 1e-3, f"all_reduce failed: got {t.item()}, expected {expected}"
    log("[PASS] all_reduce (float32)")

def test_all_gather():
    rank = dist.get_rank()
    ws = dist.get_world_size()
    t = torch.tensor([float(rank)], device="cuda")
    out = [torch.empty_like(t) for _ in range(ws)]
    dist.all_gather(out, t)
    for i, o in enumerate(out):
        assert abs(o.item() - i) < 1e-3, f"all_gather failed at rank {i}"
    log("[PASS] all_gather (float32)")

def test_all_gather_int64():
    rank = dist.get_rank()
    ws = dist.get_world_size()
    t = torch.tensor([rank], device="cuda", dtype=torch.long)
    out = [torch.empty_like(t) for _ in range(ws)]
    dist.all_gather(out, t)
    for i, o in enumerate(out):
        assert o.item() == i, f"all_gather int64 failed at rank {i}"
    log("[PASS] all_gather (int64)")

def test_alltoall_single_float_equal():
    rank = dist.get_rank()
    ws = dist.get_world_size()
    send = torch.full((ws,), float(rank), device="cuda")
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    for i in range(ws):
        assert abs(recv[i].item() - i) < 1e-3, f"a2a float equal failed: recv[{i}]={recv[i].item()}"
    log("[PASS] all_to_all_single equal-size (float32)")

def test_alltoall_single_int64_equal():
    rank = dist.get_rank()
    ws = dist.get_world_size()
    send = torch.full((ws,), rank, device="cuda", dtype=torch.long)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    for i in range(ws):
        assert recv[i].item() == i, f"a2a int64 equal failed: recv[{i}]={recv[i].item()}"
    log("[PASS] all_to_all_single equal-size (int64)")

def test_alltoall_single_bf16_equal():
    rank = dist.get_rank()
    ws = dist.get_world_size()
    send = torch.full((ws,), float(rank), device="cuda", dtype=torch.bfloat16)
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    for i in range(ws):
        assert abs(recv[i].item() - i) < 1, f"a2a bf16 equal failed: recv[{i}]={recv[i].item()}"
    log("[PASS] all_to_all_single equal-size (bfloat16)")

def test_alltoall_single_float_unequal():
    """Simulates the MoE token dispatch pattern with unequal split sizes."""
    rank = dist.get_rank()
    ws = dist.get_world_size()
    send_splits = [(rank + i + 1) for i in range(ws)]
    total_send = sum(send_splits)
    send = torch.arange(total_send, device="cuda", dtype=torch.float32)

    recv_splits_tensor = torch.tensor(send_splits, device="cuda", dtype=torch.long)
    all_splits = [torch.empty_like(recv_splits_tensor) for _ in range(ws)]
    dist.all_gather(all_splits, recv_splits_tensor)
    recv_splits = [all_splits[i][rank].item() for i in range(ws)]
    total_recv = sum(recv_splits)

    recv = torch.empty(total_recv, device="cuda", dtype=torch.float32)
    dist.all_to_all_single(recv, send,
                           output_split_sizes=recv_splits,
                           input_split_sizes=send_splits)
    log(f"[PASS] all_to_all_single unequal-size (float32), sent {total_send}, recv {total_recv}")

def test_alltoall_single_bf16_unequal():
    """Simulates the MoE token dispatch with bfloat16 (actual GSQ dtype)."""
    rank = dist.get_rank()
    ws = dist.get_world_size()
    H = 7168  # Kimi-K2.5 hidden dim
    send_splits = [(rank + i + 1) * 10 for i in range(ws)]
    total_send = sum(send_splits)
    send = torch.randn(total_send, H, device="cuda", dtype=torch.bfloat16)

    recv_splits_tensor = torch.tensor(send_splits, device="cuda", dtype=torch.long)
    all_splits = [torch.empty_like(recv_splits_tensor) for _ in range(ws)]
    dist.all_gather(all_splits, recv_splits_tensor)
    recv_splits = [all_splits[i][rank].item() for i in range(ws)]
    total_recv = sum(recv_splits)

    recv = torch.empty(total_recv, H, device="cuda", dtype=torch.bfloat16)
    dist.all_to_all_single(recv, send,
                           output_split_sizes=recv_splits,
                           input_split_sizes=send_splits)
    log(f"[PASS] all_to_all_single unequal-size bf16 ({total_send}x{H} -> {total_recv}x{H})")

def test_alltoall_single_bf16_large():
    """Large transfer mimicking actual GSQ workload: microbatch_size=2, seq=4096, top_k=8, 8 GPUs."""
    rank = dist.get_rank()
    ws = dist.get_world_size()
    H = 7168
    tokens_per_rank = 2 * 4096 * 8 // ws  # rough estimate
    send_splits = [tokens_per_rank] * ws
    total_send = sum(send_splits)
    send = torch.randn(total_send, H, device="cuda", dtype=torch.bfloat16)

    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send,
                           output_split_sizes=send_splits,
                           input_split_sizes=send_splits)
    log(f"[PASS] all_to_all_single large bf16 ({total_send}x{H})")


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(0)

    if rank == 0:
        print(f"World size: {dist.get_world_size()}", flush=True)
        print(f"NCCL version: {torch.cuda.nccl.version()}", flush=True)
        print(f"PyTorch: {torch.__version__}", flush=True)
        print(f"CUDA: {torch.version.cuda}", flush=True)
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)
        print(f"NCCL_NET: {os.environ.get('NCCL_NET', '<unset>')}", flush=True)
        print(f"NCCL_PROTO: {os.environ.get('NCCL_PROTO', '<unset>')}", flush=True)
        print("", flush=True)

    tests = [
        ("all_reduce", test_all_reduce),
        ("all_gather float32", test_all_gather),
        ("all_gather int64", test_all_gather_int64),
        ("a2a equal float32", test_alltoall_single_float_equal),
        ("a2a equal int64", test_alltoall_single_int64_equal),
        ("a2a equal bf16", test_alltoall_single_bf16_equal),
        ("a2a unequal float32", test_alltoall_single_float_unequal),
        ("a2a unequal bf16", test_alltoall_single_bf16_unequal),
        ("a2a large bf16", test_alltoall_single_bf16_large),
    ]

    for name, fn in tests:
        dist.barrier()
        try:
            fn()
        except Exception as e:
            if rank == 0:
                print(f"[FAIL] {name}: {e}", flush=True)

    dist.barrier()
    if rank == 0:
        print("\nAll tests complete.", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
