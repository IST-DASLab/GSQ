#!/usr/bin/env python3
"""Checkpoint GSVQ on a real GGUF IQ2 tensor against a dense HF tensor."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gguf_iq import GGUFIQuantStore, gguf_name_to_hf_name  # noqa: E402
from src.quantization.gsvq import FactorizedIQuantGSVQ, format_history, train_gsvq_reconstruction  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fixed-scale GSVQ reconstruction on one real GGUF IQ2 tensor."
    )
    parser.add_argument("--gguf", required=True, help="Path to a pre-quantized GGUF file.")
    parser.add_argument("--tensor", required=True, help="GGUF tensor name or HF tensor module name.")
    parser.add_argument("--hf-model", required=True, help="Dense HF model directory containing safetensors.")
    parser.add_argument("--hf-name", default="",
                        help="Dense HF tensor name. Defaults to mapped GGUF name + '.weight'.")
    parser.add_argument("--vectors", type=int, default=8192,
                        help="Number of IQuant vectors from the start of the tensor to optimize.")
    parser.add_argument("--offset", type=int, default=0,
                        help="Vector offset into the flattened tensor.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--neighbor-candidates", type=int, default=8)
    parser.add_argument("--target-candidates", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rotation-trick", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _load_hf_tensor(model_dir: str | Path, tensor_name: str) -> torch.Tensor:
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        import json as _json

        with open(index_path, "r") as f:
            weight_map = _json.load(f)["weight_map"]
        if tensor_name not in weight_map:
            raise KeyError(f"{tensor_name!r} not found in {index_path}")
        shard = model_dir / weight_map[tensor_name]
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            return f.get_tensor(tensor_name)

    single = model_dir / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt", device="cpu") as f:
            if tensor_name not in f.keys():
                raise KeyError(f"{tensor_name!r} not found in {single}")
            return f.get_tensor(tensor_name)

    raise FileNotFoundError(f"No safetensors model found under {model_dir}")


def main() -> int:
    args = parse_args()
    store = GGUFIQuantStore(args.gguf)
    decomp = store.decompose(args.tensor, device=args.device)
    hf_name = args.hf_name
    if not hf_name:
        hf_module = decomp.hf_name or gguf_name_to_hf_name(decomp.gguf_name)
        if hf_module is None:
            raise ValueError(f"Cannot map {decomp.gguf_name!r} to an HF tensor name")
        hf_name = hf_module + ".weight"

    target = _load_hf_tensor(args.hf_model, hf_name).to(device=args.device, dtype=torch.float32)
    if tuple(target.shape) != decomp.original_shape:
        raise ValueError(
            f"Dense tensor shape {tuple(target.shape)} does not match GGUF dequant shape {decomp.original_shape}"
        )

    vector_dim = decomp.vector_dim
    start = args.offset
    end = min(start + args.vectors, decomp.vector_scales.shape[0])
    if start < 0 or start >= end:
        raise ValueError(f"Bad vector window offset={args.offset}, vectors={args.vectors}")

    target_vectors = target.reshape(-1, vector_dim)[start:end]
    quantizer = FactorizedIQuantGSVQ(
        target_vectors,
        decomp.vector_scales[start:end],
        decomp.magnitude_codebook,
        decomp.sign_codebook,
        decomp.magnitude_indices[start:end],
        decomp.sign_indices[start:end],
        candidate_count=args.candidate_count,
        neighbor_candidates=args.neighbor_candidates,
        target_candidates=args.target_candidates,
        rotation_trick=args.rotation_trick,
    )
    history = train_gsvq_reconstruction(quantizer, steps=args.steps, lr=args.lr)
    decrease = history.initial_hard_mse - history.best_hard_mse
    payload = {
        "checkpoint": "gsvq_real_gguf_tensor_reconstruction_decrease",
        "gguf": os.path.abspath(args.gguf),
        "gguf_tensor": decomp.gguf_name,
        "hf_tensor": hf_name,
        "qtype": decomp.qtype_name,
        "vector_dim": vector_dim,
        "offset": start,
        "vectors": end - start,
        "steps": args.steps,
        "initial_hard_mse": history.initial_hard_mse,
        "best_hard_mse": history.best_hard_mse,
        "final_hard_mse": history.final_hard_mse,
        "absolute_decrease": decrease,
        "relative_decrease": decrease / max(history.initial_hard_mse, 1e-30),
        "passed": history.decreased,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("GSVQ real GGUF tensor checkpoint")
        print(json.dumps(payload, indent=2, sort_keys=True))
        print(format_history(history))

    if not history.decreased:
        print("ERROR: hard reconstruction MSE did not decrease", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
