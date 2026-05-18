"""Convert GSQ compressed-tensors quantized weights into Humming's native
on-disk tensor layout.

GSQ writes per-Linear in compressed-tensors `pack-quantized` format:

    <prefix>.weight_packed : int32 [N, K * storage_bits // 32]  (LSB-first)
    <prefix>.weight_scale  : float  [N, K // group_size]
    <prefix>.weight_shape  : int    [2]                          (N, K)

with symmetric storage: the packer adds `+2**(storage_bits-1)` to the signed
code before bit-packing, so the unpacked unsigned codes live in
[0, 2**storage_bits).

The Humming kernels (`humming.layer.HummingLayer`) expect a different layout:

    weight       : int32 [N, K * num_bits // 32]   (humming.ops.pack_weight)
    weight_scale : bf16/fp16 [N, K // group_size]
    zero_point   : bf16/fp16 [N, K // group_size]  (FP zero-point path)

For our pack we use the FP zero-point path: it is lossless w.r.t. the GSQ
calibration and avoids round-trip through Humming's RTN packer.

GSQ also lets the calibrated codebook use fewer levels than the storage
container (e.g. 2-bit Gumbel quantizer with 4 levels {-2,-1,0,1} can be stored
in a 4-bit CT container). We infer the effective bit width from the actual
code range and emit Humming weights at that effective width.

Upstream Humming packing logic was adapted from an internal sibling repo; cite this file for GSQ specifics.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


def _check_humming_importable() -> None:
    try:
        import humming  # noqa: F401
        from humming import ops  # noqa: F401
        from humming.schema.humming import HummingWeightSchema  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "humming is required for this packer. Install your checkout (e.g. "
            "`uv pip install -e /path/to/humming`). See `.env.example` for optional CUDA_HOME."
        ) from e


# Humming's MARLIN packer needs whole 32x32 tiles, so the total element count
# of the unsigned int code matrix must be a multiple of 1024.
HUMMING_TILE = 32 * 32


def _unpack_ct_int32(packed: torch.Tensor, storage_bits: int) -> torch.Tensor:
    """LSB-first unpack of compressed-tensors int32 along the last dim.

    Returns int32 in [0, 2**storage_bits).
    """
    mask = (1 << storage_bits) - 1
    vals_per_el = 32 // storage_bits
    shifts = (torch.arange(vals_per_el, dtype=torch.int32) * storage_bits).to(packed.device)
    expanded = (packed.unsqueeze(-1).to(torch.int64) >> shifts.to(torch.int64)) & mask
    return expanded.reshape(*packed.shape[:-1], packed.shape[-1] * vals_per_el).to(torch.int32)


def _infer_effective_bits(code_min: int, code_max: int) -> int:
    span = code_max - code_min
    if span <= 0:
        return 2
    # Smallest b such that 2**b > span.
    return max(2, (span).bit_length())


def ct_to_humming(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: torch.Tensor,
    *,
    storage_bits: int,
    group_size: int,
    symmetric: bool = True,
    target_dtype: torch.dtype = torch.bfloat16,
    force_bits: int | None = None,
    symmetric_out: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], Dict[str, Any]]:
    """Convert one GSQ-style CT bundle into Humming layout.

    Args:
        weight_packed: int32 [N, K * storage_bits // 32], LSB-first packed.
        weight_scale:  float [N, K // group_size].
        weight_shape:  int  [2] -> (N, K), as written by GSQ.
        storage_bits:  bit width of the CT container (`num_bits` from the
                       quantization_config). May be larger than the effective
                       width if GSQ over-allocated.
        group_size:    group size of the CT scales.
        symmetric:     CT symmetric flag. Always True for current GSQ.
        target_dtype:  dtype to store scales / FP zero-points as
                       (bf16 or fp16).
        force_bits:    if set, override the inferred effective bits.
        symmetric_out: if True, emit no `zero_point` tensor. The kernel applies
                       an implicit `2**(eff_bits-1)` offset (offset-binary
                       symmetric). Only valid when the GSQ codebook spans the
                       full unsigned range `[0, 2**eff_bits)` so the implicit
                       offset matches what we'd otherwise store as a constant
                       FP zero-point. The function asserts this; pass
                       `symmetric_out=False` (the default, lossless FP zp path)
                       if you are unsure.

    Returns:
        tensors:       {'weight', 'weight_scale', 'zero_point'} ready to feed
                       `HummingLayer.load_from_tensors(...)`. All CPU.
        schema_config: dict for `HummingLayer(weight_config=...)`.
        info:          diagnostics (effective bits, code range, shapes).
    """
    _check_humming_importable()
    from humming import ops

    if target_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"target_dtype must be bf16 or fp16, got {target_dtype}")
    if not symmetric:
        raise NotImplementedError(
            "GSQ humming converter currently only handles symmetric CT storage; "
            "got symmetric=False."
        )

    N, K = int(weight_shape[0]), int(weight_shape[1])

    decoded = _unpack_ct_int32(weight_packed, storage_bits)  # int32 [N, K_padded]
    decoded = decoded[:, :K].contiguous()                    # drop pack padding

    ct_offset = 1 << (storage_bits - 1)
    code_min = int(decoded.min())
    code_max = int(decoded.max())
    eff_bits = force_bits if force_bits is not None else _infer_effective_bits(code_min, code_max)

    span = code_max - code_min
    if span >= (1 << eff_bits):
        raise ValueError(
            f"code span {span} (min={code_min}, max={code_max}) exceeds target "
            f"effective bits {eff_bits}; pass force_bits=<larger> to override."
        )

    # Shift unsigned CT codes so the smallest used code becomes 0; choose an
    # FP zero-point so that (code_hum - zero_fp) * scale == (code_ct - ct_offset) * scale.
    qweight_hum_u32 = (decoded - code_min).to(torch.int32).contiguous()
    zero_fp_value = float(ct_offset - code_min)

    if symmetric_out:
        # Offset-binary symmetric: humming's kernel implicitly subtracts
        # 2**(eff_bits-1) from the unsigned code. For that to be equivalent
        # to (code_ct - ct_offset) * scale, the shifted code (code_ct - code_min)
        # must equal (code_ct - ct_offset) + 2**(eff_bits-1), i.e.,
        # ct_offset - code_min == 2**(eff_bits-1). Assert it explicitly so a
        # codebook that doesn't fill the full negative range doesn't silently
        # produce wrong values.
        expected_zp = 1 << (eff_bits - 1)
        if int(zero_fp_value) != expected_zp or zero_fp_value != float(expected_zp):
            raise ValueError(
                f"symmetric_out=True requires the codebook to span the full "
                f"unsigned range so the implicit kernel offset matches. "
                f"Got effective zero-point {zero_fp_value} but symmetric mode "
                f"requires {expected_zp} (= 2**(eff_bits-1) with eff_bits={eff_bits}). "
                f"Drop --symmetric or rerun with force_bits matching the actual "
                f"codebook span."
            )

    if qweight_hum_u32.numel() % HUMMING_TILE != 0:
        raise ValueError(
            f"humming pack_weight requires nelement divisible by 1024; got "
            f"shape {tuple(qweight_hum_u32.shape)} ({qweight_hum_u32.numel()} elements)."
        )

    packed_weight = ops.pack_weight(qweight_hum_u32.cuda(), eff_bits).cpu()

    scales_t = weight_scale.to(target_dtype).cpu().contiguous()

    tensors: Dict[str, torch.Tensor] = {
        "weight": packed_weight,
        "weight_scale": scales_t,
    }
    schema_config: Dict[str, Any] = {
        "quant_method": "humming",
        "dtype": f"uint{eff_bits}",
        "group_size": int(group_size),
        "has_zero_point": not symmetric_out,
        "is_fp_zero_point": not symmetric_out,
    }
    if not symmetric_out:
        tensors["zero_point"] = torch.full_like(scales_t, zero_fp_value, dtype=target_dtype)
    info: Dict[str, Any] = {
        "N": N,
        "K": K,
        "storage_bits": int(storage_bits),
        "effective_bits": int(eff_bits),
        "symmetric_out": bool(symmetric_out),
        "code_min": code_min,
        "code_max": code_max,
        "ct_offset": int(ct_offset),
        "zero_fp_value": zero_fp_value,
        "group_size": int(group_size),
    }
    return tensors, schema_config, info


def humming_dequantize(
    tensors: Dict[str, torch.Tensor],
    schema_config: Dict[str, Any],
) -> torch.Tensor:
    """Run Humming's own dequantizer on the packed tensors. Useful for verifying
    that the packing matches the reference fake-quant tensor."""
    _check_humming_importable()
    from humming import dtypes
    from humming.schema.humming import HummingWeightSchema

    schema = HummingWeightSchema(
        b_dtype=dtypes.DataType.from_str(schema_config["dtype"]),
        weight_scale_group_size=schema_config["group_size"],
        has_zero_point=schema_config["has_zero_point"],
        is_fp_zero_point=schema_config["is_fp_zero_point"],
    )
    cuda_tensors = {k: v.cuda() for k, v in tensors.items()}
    return schema.dequant_tensors(cuda_tensors)


def ct_dequantize_reference(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: torch.Tensor,
    *,
    storage_bits: int,
    group_size: int,
    target_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Independent dequantization of a CT bundle, matching what GSQ's load path
    produces via `PackedQuantizationCompressor.decompress`. Used as ground truth
    in round-trip tests so we don't depend on compressed-tensors internals."""
    N, K = int(weight_shape[0]), int(weight_shape[1])
    decoded = _unpack_ct_int32(weight_packed, storage_bits)[:, :K].contiguous()
    ct_offset = 1 << (storage_bits - 1)
    signed = (decoded - ct_offset).to(torch.float32)
    scale = weight_scale.to(torch.float32)
    groups = scale.shape[1]
    assert K == group_size * groups, (
        f"shape mismatch K={K} gs={group_size} groups={groups}"
    )
    scale_exp = scale.repeat_interleave(group_size, dim=1)[:, :K]
    return (signed * scale_exp).to(target_dtype).contiguous()
