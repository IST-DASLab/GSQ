"""Convert an assembled GSQ compressed-tensors checkpoint into a Humming-format
checkpoint that `humming.layer.HummingLayer.from_safetensors` can load.

Input  (from `save_model.py`):
    <in_dir>/config.json               with quantization_config.format=pack-quantized
    <in_dir>/model-*.safetensors       per-shard tensors
    <in_dir>/model.safetensors.index.json

Per-Linear inputs in safetensors:
    <prefix>.weight_packed   int32 [N, K * num_bits // 32]
    <prefix>.weight_scale    bf16/fp32 [N, K // group_size]
    <prefix>.weight_shape    int64 [2]

Output:
    <out_dir>/config.json              with quantization_config.quant_method=humming
                                       and a per-layer `dynamic` map encoding
                                       the effective bits.
    <out_dir>/model-*.safetensors      every quantized Linear rewritten as:
        <prefix>.weight         int32 [N, K * eff_bits // 32]  (Humming packed)
        <prefix>.weight_scale   bf16  [N, K // group_size]
        <prefix>.zero_point     bf16  [N, K // group_size]
    All non-quantized tensors are copied through unchanged.

Verification:
    --verify-one <regex>   pick the first matching Linear, build a HummingLayer
                           from the converted tensors + config, run a forward
                           pass against a random bf16 activation, compare to
                           `x @ W_deq.T` where W_deq is the CT dequantization
                           of the same Linear. Reports max abs / rel error.

Usage:
    . ~/local/venvs/main/bin/activate
    export CUDA_HOME=/localhome/mhelcig/venvs/main/lib/python3.12/site-packages/nvidia/cu13
    export PATH=$CUDA_HOME/bin:$PATH

    # Convert + write a new checkpoint dir:
    python convert_to_humming.py \
        --in-dir  /path/to/assembled \
        --out-dir /path/to/assembled-humming \
        --verify-one '.*\\.layers\\.0\\..*'

    # Verify only (no write):
    python convert_to_humming.py --in-dir /path/to/assembled --verify-only \
        --verify-one '.*\\.layers\\.0\\..*'
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.quantization.humming_pack import (  # noqa: E402
    ct_dequantize_reference,
    ct_to_humming,
)


CT_SUFFIXES = ("weight_packed", "weight_scale", "weight_shape")


def load_quant_config(in_dir: Path) -> Tuple[Dict, Dict]:
    """Return (full_config, weight_subconfig)."""
    cfg = json.loads((in_dir / "config.json").read_text())
    qc = cfg["quantization_config"]
    if qc["quant_method"] != "compressed-tensors":
        raise ValueError(f"expected compressed-tensors, got {qc['quant_method']}")
    if qc.get("format") != "pack-quantized":
        raise ValueError(f"expected pack-quantized format, got {qc.get('format')}")
    weight_cfg = qc["config_groups"]["group_0"]["weights"]
    return cfg, weight_cfg


def discover_layers(in_dir: Path) -> Dict[str, Dict[str, Tuple[str, str]]]:
    """Walk safetensors index, return {prefix: {suffix: (key, shard_path)}}.

    `prefix` is the Linear module name (e.g. 'model.layers.0.mlp.gate_proj').
    """
    idx_path = in_dir / "model.safetensors.index.json"
    single_path = in_dir / "model.safetensors"
    if idx_path.exists():
        wmap = json.loads(idx_path.read_text())["weight_map"]
        wmap = {k: str(in_dir / v) for k, v in wmap.items()}
    elif single_path.exists():
        with safe_open(str(single_path), framework="pt", device="cpu") as f:
            wmap = {k: str(single_path) for k in f.keys()}
    else:
        raise FileNotFoundError(f"no safetensors index/file under {in_dir}")

    layers: Dict[str, Dict[str, Tuple[str, str]]] = {}
    for key, shard in wmap.items():
        for suf in CT_SUFFIXES:
            marker = "." + suf
            if key.endswith(marker):
                prefix = key[: -len(marker)]
                layers.setdefault(prefix, {})[suf] = (key, shard)
                break
    complete = {p: m for p, m in layers.items() if set(m) == set(CT_SUFFIXES)}
    return complete


def load_layer_tensors(
    shard_map: Dict[str, Tuple[str, str]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read weight_packed/weight_scale/weight_shape from their (possibly different) shards."""
    # Group reads per shard for fewer file opens.
    by_shard: Dict[str, List[Tuple[str, str]]] = {}
    for suf, (key, shard) in shard_map.items():
        by_shard.setdefault(shard, []).append((suf, key))
    out: Dict[str, torch.Tensor] = {}
    for shard, items in by_shard.items():
        with safe_open(shard, framework="pt", device="cpu") as f:
            for suf, key in items:
                out[suf] = f.get_tensor(key)
    return out["weight_packed"], out["weight_scale"], out["weight_shape"]


def convert_layer(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: torch.Tensor,
    *,
    storage_bits: int,
    group_size: int,
    symmetric: bool,
    target_dtype: torch.dtype,
    symmetric_out: bool = False,
):
    return ct_to_humming(
        weight_packed, weight_scale, weight_shape,
        storage_bits=storage_bits, group_size=group_size, symmetric=symmetric,
        target_dtype=target_dtype, symmetric_out=symmetric_out,
    )


def write_humming_checkpoint(
    in_dir: Path,
    out_dir: Path,
    layer_map: Dict[str, Dict[str, Tuple[str, str]]],
    storage_bits: int,
    group_size: int,
    symmetric: bool,
    target_dtype: torch.dtype,
    symmetric_out: bool = False,
):
    """Rewrite the checkpoint: for each quantized Linear emit Humming tensors;
    copy all other tensors through unchanged. Preserves shard partitioning."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load original index.
    idx_path = in_dir / "model.safetensors.index.json"
    if idx_path.exists():
        wmap_orig = json.loads(idx_path.read_text())["weight_map"]
    else:
        with safe_open(str(in_dir / "model.safetensors"), framework="pt", device="cpu") as f:
            wmap_orig = {k: "model.safetensors" for k in f.keys()}

    # Map shard filename -> list of original keys.
    by_shard: Dict[str, List[str]] = {}
    for k, shard in wmap_orig.items():
        by_shard.setdefault(shard, []).append(k)

    # For each quantized prefix, derive which Humming keys replace which CT keys.
    # Humming keys are written into whichever shard held the CT weight_packed.
    quant_keys_drop_per_shard: Dict[str, set] = {}
    quant_keys_add_per_shard: Dict[str, Dict[str, torch.Tensor]] = {}
    per_layer_cfg: Dict[str, Dict] = {}
    total_eff_bits_counter: Dict[int, int] = {}

    t0 = time.perf_counter()
    progress_total = len(layer_map)
    for i, (prefix, shard_map) in enumerate(sorted(layer_map.items()), start=1):
        wp, ws, wsh = load_layer_tensors(shard_map)
        try:
            tensors, cfg, info = convert_layer(
                wp, ws, wsh,
                storage_bits=storage_bits, group_size=group_size, symmetric=symmetric,
                target_dtype=target_dtype, symmetric_out=symmetric_out,
            )
        except Exception as e:
            raise RuntimeError(f"failed to convert {prefix}: {e}") from e

        per_layer_cfg[prefix] = cfg
        total_eff_bits_counter[info["effective_bits"]] = \
            total_eff_bits_counter.get(info["effective_bits"], 0) + 1

        # Target shard = the shard that held the CT weight_packed for this prefix.
        _, target_shard = shard_map["weight_packed"]
        target_shard_name = Path(target_shard).name
        # Drop the three CT keys.
        for suf in CT_SUFFIXES:
            key = f"{prefix}.{suf}"
            src_shard = Path(shard_map[suf][1]).name
            quant_keys_drop_per_shard.setdefault(src_shard, set()).add(key)
        # Add the Humming keys into target_shard. zero_point is omitted in
        # symmetric_out mode.
        adds = quant_keys_add_per_shard.setdefault(target_shard_name, {})
        adds[f"{prefix}.weight"] = tensors["weight"]
        adds[f"{prefix}.weight_scale"] = tensors["weight_scale"]
        if "zero_point" in tensors:
            adds[f"{prefix}.zero_point"] = tensors["zero_point"]

        if i % 20 == 0 or i == progress_total:
            dt = time.perf_counter() - t0
            print(f"  [{i}/{progress_total}] {prefix} -> uint{info['effective_bits']}  "
                  f"({dt:.1f}s, {i / max(dt, 1e-6):.1f} layers/s)", flush=True)

    print(f"\neffective-bit histogram: {total_eff_bits_counter}")

    # Now stream-write each shard with the modifications.
    new_wmap: Dict[str, str] = {}
    total_bytes = 0
    print(f"\nwriting {len(by_shard)} shards to {out_dir}...")
    for shard_name in sorted({Path(s).name for s in by_shard}):
        src_path = in_dir / shard_name
        drop = quant_keys_drop_per_shard.get(shard_name, set())
        adds = quant_keys_add_per_shard.get(shard_name, {})

        tensors_out: Dict[str, torch.Tensor] = {}
        with safe_open(str(src_path), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in drop:
                    continue
                tensors_out[k] = f.get_tensor(k)
        tensors_out.update(adds)
        for k, v in tensors_out.items():
            tensors_out[k] = v.contiguous()
            total_bytes += tensors_out[k].numel() * tensors_out[k].element_size()
            new_wmap[k] = shard_name
        out_path = out_dir / shard_name
        save_file(tensors_out, str(out_path))
        print(f"  {shard_name}: {len(tensors_out)} tensors")

    # Write a new index.
    if idx_path.exists() or len(by_shard) > 1:
        new_index = {"metadata": {"total_size": int(total_bytes)}, "weight_map": new_wmap}
        (out_dir / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2))

    # Copy tokenizer / generation_config / etc.
    keep_files = [
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "tokenizer.model", "tiktoken.model",
        "special_tokens_map.json", "generation_config.json", "preprocessor_config.json",
    ]
    for fn in os.listdir(in_dir):
        src = in_dir / fn
        if not src.is_file():
            continue
        if fn in keep_files or fn.endswith(".py"):
            shutil.copy2(src, out_dir / fn)

    # Write the updated config.json with humming quantization_config.
    cfg_full = json.loads((in_dir / "config.json").read_text())
    old_qc = cfg_full["quantization_config"]
    ignore = old_qc.get("ignore", [])

    # Build a `dynamic` map: regex -> per-layer humming config. Use the layer
    # prefix as an exact-match regex (escaped). The HummingLayer.from_safetensors
    # treats `regex[2:]` as the actual pattern (the first two chars are a flag).
    # HummingLayer.from_safetensors parses entries as:
    #   regex[0] != "-"  → assert prefix does NOT match (exclusion)
    #   regex[0] == "-"  → if match, override config with the linked entry
    # So we use "-:" for per-layer overrides.
    dynamic: Dict[str, Dict] = {}
    for prefix, cfg_l in per_layer_cfg.items():
        pat = "-:" + re.escape(prefix) + "$"
        dynamic[pat] = cfg_l

    # Pick a "default" config from the most-common bitwidth so the top-level
    # config_groups entry is at least sensible. HummingLayer.from_safetensors
    # uses `config_groups` when `dynamic` doesn't match; the per-layer dynamic
    # entries take precedence.
    most_common_bits = max(total_eff_bits_counter, key=total_eff_bits_counter.get)
    top_cfg = {
        "config_groups": {
            "group_0": {
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "dtype": f"uint{most_common_bits}",
                    "group_size": group_size,
                    "has_zero_point": not symmetric_out,
                    "is_fp_zero_point": not symmetric_out,
                    "num_bits": most_common_bits,
                    "strategy": "group",
                    "symmetric": bool(symmetric_out),
                    "type": "int",
                },
            }
        },
        "format": "pack-quantized",
        "ignore": ignore,
        "quant_method": "humming",
        "quantization_status": "compressed",
        "dynamic": dynamic,
    }
    cfg_full["quantization_config"] = top_cfg
    (out_dir / "config.json").write_text(json.dumps(cfg_full, indent=2))

    print(f"\nwrote {len(per_layer_cfg)} quantized layers, "
          f"~{total_bytes / 1024**3:.2f} GB total")
    print(f"output: {out_dir}")


def verify_one(
    layer_map: Dict[str, Dict[str, Tuple[str, str]]],
    pattern: str,
    storage_bits: int,
    group_size: int,
    symmetric: bool,
    target_dtype: torch.dtype,
    symmetric_out: bool = False,
):
    """Pick the first Linear matching `pattern`, convert it, run a kernel
    forward, and compare to the CT-dequant matmul reference."""
    pat = re.compile(pattern)
    matches = [p for p in sorted(layer_map) if pat.search(p)]
    if not matches:
        raise SystemExit(f"no layer matched verify-one pattern {pattern!r}; "
                         f"first few layers: {sorted(layer_map)[:5]}")
    name = matches[0]
    print(f"verifying {name}  (symmetric_out={symmetric_out})")
    wp, ws, wsh = load_layer_tensors(layer_map[name])
    ref = ct_dequantize_reference(
        wp, ws, wsh, storage_bits=storage_bits, group_size=group_size,
        target_dtype=target_dtype,
    )
    tensors, cfg, info = ct_to_humming(
        wp, ws, wsh,
        storage_bits=storage_bits, group_size=group_size, symmetric=symmetric,
        target_dtype=target_dtype, symmetric_out=symmetric_out,
    )
    print(f"  info: {info}")
    print(f"  schema: {cfg}")

    # Build kernel layer and forward.
    from humming import dtypes
    from humming.layer import HummingLayer
    from humming.schema.humming import HummingWeightSchema

    schema = HummingWeightSchema(
        b_dtype=dtypes.DataType.from_str(cfg["dtype"]),
        weight_scale_group_size=cfg["group_size"],
        has_zero_point=cfg["has_zero_point"],
        is_fp_zero_point=cfg["is_fp_zero_point"],
    )
    N, K = info["N"], info["K"]
    device = "cuda"
    layer = HummingLayer(
        shape_n=N, shape_k=K, weight_config=schema, torch_dtype=target_dtype,
    ).to(device)
    layer.load_from_tensors({k: v.to(device) for k, v in tensors.items()})
    layer.transform()

    torch.manual_seed(0)
    x = (torch.randn(16, K, dtype=target_dtype, device=device) * 0.05)
    y_hum = layer(x)
    y_ref = x @ ref.to(device).t()
    diff = (y_hum.float() - y_ref.float()).abs()
    rel = diff.max().item() / max(y_ref.float().abs().max().item(), 1e-6)
    print(f"  forward: max_abs={diff.max().item():.3e}  "
          f"mean={diff.mean().item():.3e}  rel_max={rel:.3e}")
    if rel > 5e-2:
        raise SystemExit(f"forward error too large: rel_max={rel}")
    print("  OK")


def parse_args():
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", required=True, help="Path to assembled GSQ checkpoint dir.")
    p.add_argument("--out-dir", default=None, help="Output dir for converted humming checkpoint.")
    p.add_argument("--verify-only", action="store_true",
                   help="Only verify, don't write anything.")
    p.add_argument("--verify-one", default=None,
                   help="Regex; verify the first matching Linear by running a kernel forward "
                        "pass and comparing to the dequant reference.")
    p.add_argument("--target-dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--symmetric", action="store_true",
                   help="Emit Humming's offset-binary symmetric format: no per-layer "
                        "zero_point tensor; the kernel applies an implicit "
                        "2^(eff_bits-1) offset. Requires the GSQ codebook to span the "
                        "full unsigned range (true for current 2/3/4-bit Gumbel quantizers).")
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.in_dir).resolve()
    target_dtype = getattr(torch, args.target_dtype)

    full_cfg, weight_cfg = load_quant_config(in_dir)
    storage_bits = int(weight_cfg["num_bits"])
    group_size = int(weight_cfg["group_size"])
    symmetric = bool(weight_cfg["symmetric"])
    print(f"input checkpoint: {in_dir}")
    print(f"  storage_bits={storage_bits}  group_size={group_size}  symmetric={symmetric}")

    layer_map = discover_layers(in_dir)
    print(f"  found {len(layer_map)} quantized Linear modules")

    if args.verify_one:
        verify_one(layer_map, args.verify_one,
                   storage_bits=storage_bits, group_size=group_size, symmetric=symmetric,
                   target_dtype=target_dtype, symmetric_out=args.symmetric)

    if args.verify_only:
        return

    if args.out_dir is None:
        raise SystemExit("pass --out-dir (or --verify-only)")
    out_dir = Path(args.out_dir).resolve()
    write_humming_checkpoint(
        in_dir, out_dir, layer_map,
        storage_bits=storage_bits, group_size=group_size, symmetric=symmetric,
        target_dtype=target_dtype, symmetric_out=args.symmetric,
    )


if __name__ == "__main__":
    main()
