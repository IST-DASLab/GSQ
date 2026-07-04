import torch

from src.distributed.dense_mlp_tp import DenseMLPTensorParallel


def test_gate_proj_shards_rows():
    Q = torch.arange(16 * 8).reshape(16, 8)
    scales = torch.arange(16 * 2).reshape(16, 2)

    tp0 = DenseMLPTensorParallel(rank=0, world_size=2, groupsize=4)
    q0, s0, spec0 = tp0.shard_quantizer_inputs("layer.mlp.gate_proj", Q, scales)

    tp1 = DenseMLPTensorParallel(rank=1, world_size=2, groupsize=4)
    q1, s1, spec1 = tp1.shard_quantizer_inputs("layer.mlp.gate_proj", Q, scales)

    assert spec0.shard_dim == spec1.shard_dim == 0
    assert torch.equal(torch.cat([q0, q1], dim=0), Q)
    assert torch.equal(torch.cat([s0, s1], dim=0), scales)


def test_down_proj_shards_group_aligned_columns():
    Q = torch.arange(8 * 16).reshape(8, 16)
    scales = torch.arange(8 * 4).reshape(8, 4)

    tp0 = DenseMLPTensorParallel(rank=0, world_size=2, groupsize=4)
    q0, s0, spec0 = tp0.shard_quantizer_inputs("layer.mlp.down_proj", Q, scales)

    tp1 = DenseMLPTensorParallel(rank=1, world_size=2, groupsize=4)
    q1, s1, spec1 = tp1.shard_quantizer_inputs("layer.mlp.down_proj", Q, scales)

    assert spec0.shard_dim == spec1.shard_dim == 1
    assert torch.equal(torch.cat([q0, q1], dim=1), Q)
    assert torch.equal(torch.cat([s0, s1], dim=1), scales)


def test_down_proj_rejects_group_misalignment():
    Q = torch.zeros(8, 12)
    scales = torch.zeros(8, 3)
    tp = DenseMLPTensorParallel(rank=0, world_size=2, groupsize=4)

    try:
        tp.shard_quantizer_inputs("layer.mlp.down_proj", Q, scales)
    except ValueError as exc:
        assert "aligned" in str(exc)
    else:
        raise AssertionError("Expected group-alignment validation to fail")
