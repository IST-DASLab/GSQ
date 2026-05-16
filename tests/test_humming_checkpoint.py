"""End-to-end check: pick a Linear from a converted humming checkpoint, load
it via `HummingLayer.from_safetensors(name, prefix=...)`, run a kernel forward,
and compare to a dequantized reference loaded from the original GSQ
compressed-tensors checkpoint.

This exercises:
  - the config.json `quantization_config` we write (top-level + per-layer
    dynamic regex overrides)
  - the safetensors shard layout (weight / weight_scale / zero_point per prefix)
  - the kernel forward against a known-good reference

Usage:
    . ~/local/venvs/main/bin/activate
    export CUDA_HOME=/localhome/mhelcig/venvs/main/lib/python3.12/site-packages/nvidia/cu13
    export PATH=$CUDA_HOME/bin:$PATH

    python tests/test_humming_checkpoint.py \
        --ct-dir       /path/to/assembled-ct \
        --humming-dir  /path/to/assembled-humming \
        --layer-pattern '.*\\.layers\\.0\\.mlp\\.gate_proj'
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.quantization.humming_pack import ct_dequantize_reference  # noqa: E402


def find_layer_in_ct(ct_dir: Path, pattern: str):
    idx_path = ct_dir / "model.safetensors.index.json"
    if idx_path.exists():
        wmap = json.loads(idx_path.read_text())["weight_map"]
    else:
        with safe_open(str(ct_dir / "model.safetensors"), framework="pt", device="cpu") as f:
            wmap = {k: "model.safetensors" for k in f.keys()}
    pat = re.compile(pattern)
    prefixes = set()
    for k in wmap:
        if k.endswith(".weight_packed"):
            p = k[: -len(".weight_packed")]
            if pat.search(p):
                prefixes.add(p)
    return sorted(prefixes), wmap


def load_ct_reference(ct_dir: Path, prefix: str, storage_bits: int, group_size: int, target_dtype):
    idx_path = ct_dir / "model.safetensors.index.json"
    if idx_path.exists():
        wmap = json.loads(idx_path.read_text())["weight_map"]
    else:
        with safe_open(str(ct_dir / "model.safetensors"), framework="pt", device="cpu") as f:
            wmap = {k: "model.safetensors" for k in f.keys()}
    tensors = {}
    by_shard = {}
    for suf in ("weight_packed", "weight_scale", "weight_shape"):
        key = f"{prefix}.{suf}"
        shard = wmap[key]
        by_shard.setdefault(shard, []).append((suf, key))
    for shard, items in by_shard.items():
        with safe_open(str(ct_dir / shard), framework="pt", device="cpu") as f:
            for suf, key in items:
                tensors[suf] = f.get_tensor(key)
    return ct_dequantize_reference(
        tensors["weight_packed"], tensors["weight_scale"], tensors["weight_shape"],
        storage_bits=storage_bits, group_size=group_size, target_dtype=target_dtype,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ct-dir", required=True)
    p.add_argument("--humming-dir", required=True)
    p.add_argument("--layer-pattern", required=True)
    p.add_argument("--target-dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = p.parse_args()

    ct_dir = Path(args.ct_dir).resolve()
    hum_dir = Path(args.humming_dir).resolve()
    target_dtype = getattr(torch, args.target_dtype)

    # Read storage_bits / group_size from the *original* CT checkpoint so we
    # can dequantize the reference correctly.
    ct_cfg = json.loads((ct_dir / "config.json").read_text())
    wcfg = ct_cfg["quantization_config"]["config_groups"]["group_0"]["weights"]
    storage_bits = int(wcfg["num_bits"])
    group_size = int(wcfg["group_size"])

    prefixes, _ = find_layer_in_ct(ct_dir, args.layer_pattern)
    if not prefixes:
        raise SystemExit(f"no layer matched {args.layer_pattern!r} in {ct_dir}")
    prefix = prefixes[0]
    print(f"testing layer: {prefix}")

    ref = load_ct_reference(ct_dir, prefix, storage_bits, group_size, target_dtype).cuda()
    N, K = ref.shape
    print(f"  N={N} K={K} target_dtype={target_dtype}")

    # Load the humming layer via from_safetensors. This is the path that
    # production code would use.
    from humming.layer import HummingLayer

    layer = HummingLayer.from_safetensors(
        str(hum_dir), prefix=prefix, torch_dtype=target_dtype,
    ).cuda()
    layer.transform()

    torch.manual_seed(0)
    x = torch.randn(16, K, dtype=target_dtype, device="cuda") * 0.05
    y_hum = layer(x)
    y_ref = x @ ref.t()
    diff = (y_hum.float() - y_ref.float()).abs()
    rel = diff.max().item() / max(y_ref.float().abs().max().item(), 1e-6)
    print(f"  forward: max_abs={diff.max().item():.3e}  mean={diff.mean().item():.3e}  "
          f"rel_max={rel:.3e}")
    assert rel < 5e-2, f"forward error too large: rel_max={rel}"
    print("  OK")


if __name__ == "__main__":
    main()
