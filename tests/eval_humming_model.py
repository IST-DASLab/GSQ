"""Full-model end-to-end check for the GSQ -> humming conversion.

Loads Qwen3-0.6B twice:
  A) Reference: every quantized Linear is materialized in bf16 by
     `PackedQuantizationCompressor.decompress` on the CT checkpoint.
  B) Humming: every quantized Linear is replaced with a `HummingLayer` loaded
     from the converted humming checkpoint (kernel forward).

Both models share the same non-quantized weights (embeddings, attention,
layernorms, lm_head) loaded from the CT checkpoint.

Tests:
  - Logit forward on a small batch; report max abs / rel error.
  - Greedy 32-token generation from a fixed prompt; compare token sequences.
  - Wikitext-2 PPL on a small slice for both models.

Usage:
    . ~/local/venvs/main/bin/activate
    export CUDA_HOME=/localhome/mhelcig/venvs/main/lib/python3.12/site-packages/nvidia/cu13
    export PATH=$CUDA_HOME/bin:$PATH

    python tests/eval_humming_model.py \
        --ct-dir      /path/to/assembled-ct \
        --humming-dir /path/to/assembled-humming \
        --base-model  Qwen/Qwen3-0.6B \
        --ppl-tokens  8192
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ----- module replacement helpers -----------------------------------------

class HummingLinearAdapter(nn.Module):
    """Wrap a HummingLayer so it can drop in for an `nn.Linear` whose input
    is [..., in_features]."""

    def __init__(self, hum_layer, in_features: int, out_features: int, bias=None):
        super().__init__()
        self.hum_layer = hum_layer
        self.in_features = in_features
        self.out_features = out_features
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features).contiguous()
        y = self.hum_layer(x_flat)
        y = y.reshape(*orig_shape[:-1], self.out_features)
        if self.bias is not None:
            y = y + self.bias
        return y


def walk_named_modules(model: nn.Module):
    for name, mod in model.named_modules():
        yield name, mod


def find_quantized_linear_prefixes(ct_dir: Path) -> List[str]:
    idx = ct_dir / "model.safetensors.index.json"
    if idx.exists():
        wmap = json.loads(idx.read_text())["weight_map"]
    else:
        with safe_open(str(ct_dir / "model.safetensors"), framework="pt", device="cpu") as f:
            wmap = {k: "model.safetensors" for k in f.keys()}
    out = set()
    for k in wmap:
        if k.endswith(".weight_packed"):
            out.add(k[: -len(".weight_packed")])
    return sorted(out)


def get_module_by_name(root: nn.Module, dotted: str) -> nn.Module:
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def set_module_by_name(root: nn.Module, dotted: str, replacement: nn.Module) -> None:
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], replacement)


# ----- non-quantized weight loading ---------------------------------------

def load_non_quantized_weights(model: nn.Module, ct_dir: Path, dtype: torch.dtype, device: str):
    """Load every tensor in the CT checkpoint that isn't a packed-quantization
    artifact directly onto the model. Quantized Linears are handled separately."""
    idx = ct_dir / "model.safetensors.index.json"
    if idx.exists():
        wmap = json.loads(idx.read_text())["weight_map"]
    else:
        with safe_open(str(ct_dir / "model.safetensors"), framework="pt", device="cpu") as f:
            wmap = {k: "model.safetensors" for k in f.keys()}

    quant_suffixes = (".weight_packed", ".weight_scale", ".weight_shape")
    by_shard: Dict[str, List[str]] = {}
    for k, shard in wmap.items():
        if any(k.endswith(s) for s in quant_suffixes):
            continue
        by_shard.setdefault(shard, []).append(k)

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    for shard, keys in by_shard.items():
        with safe_open(str(ct_dir / shard), framework="pt", device="cpu") as f:
            for k in keys:
                if k.endswith("inv_freq"):
                    continue
                if k not in params and k not in buffers:
                    continue
                t = f.get_tensor(k).to(dtype=dtype)
                set_module_tensor_to_device(model, k, device, value=t, dtype=dtype)


# ----- reference path (CT decompress) -------------------------------------

def install_ct_decompressed_linears(
    model: nn.Module, ct_dir: Path, quant_prefixes: List[str], dtype: torch.dtype, device: str,
):
    """For each quantized Linear, decompress the CT packed tensor into a dense
    bf16 weight and assign to the original `nn.Linear.weight`."""
    from compressed_tensors import PackedQuantizationCompressor
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationScheme,
        QuantizationStrategy,
        QuantizationType,
    )

    cfg = json.loads((ct_dir / "config.json").read_text())
    wc = cfg["quantization_config"]["config_groups"]["group_0"]["weights"]
    qargs = QuantizationArgs(
        num_bits=int(wc["num_bits"]),
        type=QuantizationType.INT,
        symmetric=bool(wc["symmetric"]),
        group_size=int(wc["group_size"]),
        strategy=QuantizationStrategy.GROUP,
    )
    scheme = QuantizationScheme(targets=["Linear"], weights=qargs)
    compressor = PackedQuantizationCompressor()

    idx = ct_dir / "model.safetensors.index.json"
    if idx.exists():
        wmap = json.loads(idx.read_text())["weight_map"]
    else:
        with safe_open(str(ct_dir / "model.safetensors"), framework="pt", device="cpu") as f:
            wmap = {k: "model.safetensors" for k in f.keys()}

    by_shard: Dict[str, List[str]] = {}
    for prefix in quant_prefixes:
        for suf in ("weight_packed", "weight_scale", "weight_shape"):
            k = f"{prefix}.{suf}"
            by_shard.setdefault(wmap[k], []).append(k)

    raw: Dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(str(ct_dir / shard), framework="pt", device="cpu") as f:
            for k in keys:
                raw[k] = f.get_tensor(k)

    for prefix in quant_prefixes:
        comp = {
            "weight_packed": raw[f"{prefix}.weight_packed"],
            "weight_scale":  raw[f"{prefix}.weight_scale"],
            "weight_shape":  raw[f"{prefix}.weight_shape"],
        }
        deq = compressor.decompress({k: v for k, v in comp.items()}, scheme)
        W = deq["weight"].to(dtype=dtype)
        # set_module_tensor_to_device wants the full param name.
        set_module_tensor_to_device(model, f"{prefix}.weight", device, value=W, dtype=dtype)


# ----- humming path -------------------------------------------------------

def install_humming_linears(
    model: nn.Module, hum_dir: Path, quant_prefixes: List[str], dtype: torch.dtype, device: str,
):
    """For each quantized Linear, build a `HummingLayer` from the humming
    checkpoint and swap it for the original `nn.Linear`."""
    from humming.layer import HummingLayer

    for prefix in quant_prefixes:
        orig = get_module_by_name(model, prefix)
        assert isinstance(orig, nn.Linear), f"{prefix} is {type(orig)}, expected nn.Linear"
        in_features = orig.in_features
        out_features = orig.out_features
        bias = None
        if orig.bias is not None and orig.bias.device.type != "meta":
            bias = orig.bias.detach().to(dtype=dtype, device=device)
        layer = HummingLayer.from_safetensors(
            str(hum_dir), prefix=prefix, torch_dtype=dtype,
        ).to(device)
        layer.transform()
        adapter = HummingLinearAdapter(layer, in_features, out_features, bias=bias).to(device)
        set_module_by_name(model, prefix, adapter)


# ----- model builder ------------------------------------------------------

def build_model(base_model_name: str, dtype: torch.dtype, attn_impl: str = "sdpa"):
    cfg = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, attn_implementation=attn_impl, trust_remote_code=True).eval()
    return model, cfg


# ----- tests --------------------------------------------------------------

def run_logit_compare(
    base_model_name: str, ct_dir: Path, hum_dir: Path,
    dtype: torch.dtype, device: str,
):
    print("\n=== logit forward comparison ===")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    prompt = "The capital of France is Paris. The capital of Germany is"
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    quant_prefixes = find_quantized_linear_prefixes(ct_dir)
    print(f"  {len(quant_prefixes)} quantized linears")

    # Reference model.
    print("  building reference (CT-decompressed) model...")
    ref_model, _ = build_model(base_model_name, dtype)
    load_non_quantized_weights(ref_model, ct_dir, dtype, device)
    install_ct_decompressed_linears(ref_model, ct_dir, quant_prefixes, dtype, device)
    ref_model.to(device).eval()
    with torch.no_grad():
        ref_logits = ref_model(ids).logits.float()

    # Free ref to make room.
    del ref_model
    torch.cuda.empty_cache()

    # Humming model.
    print("  building humming (kernel) model...")
    hum_model, _ = build_model(base_model_name, dtype)
    load_non_quantized_weights(hum_model, ct_dir, dtype, device)
    install_humming_linears(hum_model, hum_dir, quant_prefixes, dtype, device)
    hum_model.to(device).eval()
    with torch.no_grad():
        hum_logits = hum_model(ids).logits.float()

    diff = (hum_logits - ref_logits).abs()
    rel = diff.max().item() / max(ref_logits.abs().max().item(), 1e-6)
    print(f"  logit max_abs={diff.max().item():.3e}  mean_abs={diff.mean().item():.3e}  "
          f"rel_max={rel:.3e}")

    # Argmax token agreement at every position.
    ref_top1 = ref_logits.argmax(dim=-1)
    hum_top1 = hum_logits.argmax(dim=-1)
    same = (ref_top1 == hum_top1).float().mean().item()
    print(f"  top-1 token agreement across {ref_top1.numel()} positions: {same*100:.1f}%")

    return hum_model, tokenizer, quant_prefixes, ref_logits, hum_logits


def run_generation(model, tokenizer, device: str, prompt: str, n_new: int = 32):
    print(f"\n=== greedy generation: {n_new} tokens ===")
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=n_new, do_sample=False, temperature=None, top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  prompt: {prompt!r}")
    print(f"  output: {text!r}")
    return out


def run_ppl(
    base_model_name: str, ct_dir: Path, hum_dir: Path,
    dtype: torch.dtype, device: str, n_tokens: int = 8192, stride: int = 512,
):
    print(f"\n=== Wikitext-2 PPL (~{n_tokens} tokens, stride {stride}) ===")
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids[0][:n_tokens].unsqueeze(0).to(device)

    def ppl_for(model):
        nlls = []
        total = 0
        for i in range(0, enc.size(1) - 1, stride):
            chunk = enc[:, i : i + stride + 1]
            if chunk.size(1) < 2:
                break
            with torch.no_grad():
                logits = model(chunk[:, :-1]).logits.float()
            target = chunk[:, 1:]
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="sum",
            )
            nlls.append(loss.item())
            total += target.numel()
        return torch.exp(torch.tensor(sum(nlls) / total)).item()

    quant_prefixes = find_quantized_linear_prefixes(ct_dir)

    print("  loading reference model (CT-decompressed)...")
    ref_model, _ = build_model(base_model_name, dtype)
    load_non_quantized_weights(ref_model, ct_dir, dtype, device)
    install_ct_decompressed_linears(ref_model, ct_dir, quant_prefixes, dtype, device)
    ref_model.to(device).eval()
    t0 = time.time()
    ref_ppl = ppl_for(ref_model)
    print(f"  reference ppl = {ref_ppl:.4f}  ({time.time()-t0:.1f}s)")
    del ref_model
    torch.cuda.empty_cache()

    print("  loading humming model (kernel)...")
    hum_model, _ = build_model(base_model_name, dtype)
    load_non_quantized_weights(hum_model, ct_dir, dtype, device)
    install_humming_linears(hum_model, hum_dir, quant_prefixes, dtype, device)
    hum_model.to(device).eval()
    t0 = time.time()
    hum_ppl = ppl_for(hum_model)
    print(f"  humming   ppl = {hum_ppl:.4f}  ({time.time()-t0:.1f}s)")

    rel = abs(hum_ppl - ref_ppl) / max(ref_ppl, 1e-6)
    print(f"  |Δppl|/ppl = {rel:.3e}")
    return ref_ppl, hum_ppl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ct-dir", required=True)
    p.add_argument("--humming-dir", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--ppl-tokens", type=int, default=8192)
    p.add_argument("--ppl-stride", type=int, default=512)
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument("--skip-gen", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda"
    dtype = getattr(torch, args.dtype)
    ct_dir = Path(args.ct_dir).resolve()
    hum_dir = Path(args.humming_dir).resolve()

    hum_model, tokenizer, quant_prefixes, _, _ = run_logit_compare(
        args.base_model, ct_dir, hum_dir, dtype, device,
    )

    if not args.skip_gen:
        run_generation(hum_model, tokenizer, device,
                       "The capital of France is Paris. The capital of Germany is", n_new=32)

    del hum_model
    torch.cuda.empty_cache()

    if not args.skip_ppl:
        run_ppl(args.base_model, ct_dir, hum_dir, dtype, device,
                n_tokens=args.ppl_tokens, stride=args.ppl_stride)


if __name__ == "__main__":
    main()
