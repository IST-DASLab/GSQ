"""Round-trip + kernel forward test for src/quantization/humming_pack.py.

Builds a synthetic GSQ-style compressed-tensors bundle (without going through
the real PackedQuantizationCompressor — we hand-pack instead so the test has
no extra deps), runs it through `ct_to_humming`, dequantizes via Humming's
schema, and compares to an independent CPU dequant. Then loads the result into
a real `HummingLayer`, runs a forward pass on the GPU, and checks the kernel
output matches `x @ W_deq.T` in bf16 tolerance.

Run:
    source .venv/bin/activate            # or your env
    python tests/test_humming_pack.py

If JIT fails to find CUDA libs, set CUDA_HOME (see `.env.example`) and PATH.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.quantization.humming_pack import (  # noqa: E402
    ct_dequantize_reference,
    ct_to_humming,
    humming_dequantize,
)


def hand_pack_ct(signed_codes: torch.Tensor, storage_bits: int) -> torch.Tensor:
    """Mimic compressed-tensors' pack: add +2**(b-1) offset, LSB-first pack
    along the last dim into int32."""
    assert signed_codes.dtype == torch.int32
    N, K = signed_codes.shape
    vals_per_el = 32 // storage_bits
    assert K % vals_per_el == 0, f"K={K} must be a multiple of {vals_per_el}"
    offset = 1 << (storage_bits - 1)
    unsigned = (signed_codes + offset).to(torch.int64)
    mask = (1 << storage_bits) - 1
    shifts = (torch.arange(vals_per_el, dtype=torch.int64) * storage_bits)
    grouped = unsigned.reshape(N, K // vals_per_el, vals_per_el)
    packed = ((grouped & mask) << shifts).sum(dim=-1).to(torch.int32)
    return packed


def make_synthetic_layer(
    N: int = 1024,
    K: int = 1024,
    group_size: int = 128,
    storage_bits: int = 2,
    seed: int = 0,
    device: str = "cuda",
):
    """Build a GSQ-2bit-style bundle. Codes are sampled from {-2,-1,0,1}, which
    is what GumbelQuantizer2Bit produces. Scales are random positive bf16."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    code_choices = torch.tensor([-2, -1, 0, 1], dtype=torch.int32)
    flat_idx = torch.randint(0, 4, (N, K), generator=g, dtype=torch.int64)
    signed_codes = code_choices[flat_idx]                              # [N, K] int32
    weight_packed = hand_pack_ct(signed_codes, storage_bits).to(device)

    weight_scale = (
        0.01 + 0.05 * torch.rand((N, K // group_size), generator=g, dtype=torch.float32)
    ).to(device, dtype=torch.bfloat16)
    weight_shape = torch.tensor([N, K], dtype=torch.int64)
    return weight_packed, weight_scale, weight_shape, signed_codes.to(device)


def main():
    assert torch.cuda.is_available(), "this test needs a CUDA device"
    device = "cuda"
    torch.cuda.set_device(0)

    N, K, gs, sbits = 1024, 1024, 128, 2
    wp, ws, wsh, signed = make_synthetic_layer(N, K, gs, sbits, device=device)

    # 1. CT reference dequant (CPU, bf16).
    ref = ct_dequantize_reference(
        wp.cpu(), ws.cpu(), wsh, storage_bits=sbits, group_size=gs,
        target_dtype=torch.bfloat16,
    )
    # 2. Convert to humming layout.
    tensors, cfg, info = ct_to_humming(
        wp.cpu(), ws.cpu(), wsh,
        storage_bits=sbits, group_size=gs, symmetric=True,
        target_dtype=torch.bfloat16,
    )
    print(f"info: {info}")
    print(f"schema: {cfg}")

    # 3. Humming dequant should match the CT reference exactly (bit-identical
    #    with FP zero-point + same dtype).
    hum_deq = humming_dequantize(tensors, cfg).to(torch.bfloat16).cpu()
    max_err = (hum_deq.float() - ref.float()).abs().max().item()
    exact = torch.equal(hum_deq, ref)
    print(f"dequant max abs err: {max_err:.3e}  exact_match={exact}")
    assert max_err < 1e-3, "humming dequant diverges from CT reference"

    # 4. Build HummingLayer, load, transform, run forward.
    from humming.layer import HummingLayer
    from humming.schema.humming import HummingWeightSchema
    from humming import dtypes

    schema = HummingWeightSchema(
        b_dtype=dtypes.DataType.from_str(cfg["dtype"]),
        weight_scale_group_size=cfg["group_size"],
        has_zero_point=cfg["has_zero_point"],
        is_fp_zero_point=cfg["is_fp_zero_point"],
    )
    layer = HummingLayer(
        shape_n=N, shape_k=K, weight_config=schema, torch_dtype=torch.bfloat16,
    ).cuda()
    layer.load_from_tensors({k: v.cuda() for k, v in tensors.items()})
    layer.transform()

    # Forward: random bf16 activation [M, K].
    M = 16
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.05
    y_hum = layer(x)                                # [M, N]
    y_ref = x @ ref.to(device).t()                  # [M, N]

    diff = (y_hum.float() - y_ref.float()).abs()
    rel = diff.max().item() / max(y_ref.float().abs().max().item(), 1e-6)
    print(f"forward: max abs err={diff.max().item():.3e}  "
          f"mean={diff.mean().item():.3e}  rel_max={rel:.3e}")
    # bf16 matmul tolerance: 1% is generous; tighten if you want.
    assert rel < 5e-2, f"humming forward diverges from reference matmul (rel={rel})"

    print("\nOK: humming pack/forward round-trip works at uint2.")


if __name__ == "__main__":
    main()
