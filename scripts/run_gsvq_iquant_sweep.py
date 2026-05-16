#!/usr/bin/env python3
"""Run fixed-scale GSVQ reconstruction sweeps over GGUF IQuant tensors."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gguf_iq import GGUFIQuantStore, gguf_name_to_hf_name  # noqa: E402
from src.quantization.gsvq import (  # noqa: E402
    FactorizedIQuantGSVQ,
    PairedMagnitudeIQuantGSVQ,
    train_gsvq_reconstruction,
)


PRESETS = {
    "original": {
        "steps": 10,
        "lr": 1e-4,
        "weight_decay": 1.0,
        "temperature": (2.0, 0.05),
        "logit_scale": (100.0, 500.0),
        "candidate_count": 16,
        "neighbor_candidates": 8,
        "target_candidates": 8,
        "std": 0.01,
        "strength": 6.0,
        "optimizer_name": "lion",
        "grad_clip": 0.0,
        "restarts": 1,
    },
    "tuned": {
        "steps": 20,
        "lr": 0.05,
        "weight_decay": 0.0,
        "temperature": (1.5, 0.1),
        "logit_scale": (1.0, 40.0),
        "candidate_count": 16,
        "neighbor_candidates": 8,
        "target_candidates": 8,
        "std": 0.01,
        "strength": 0.25,
        "optimizer_name": "adamw",
        "grad_clip": 0.0,
        "restarts": 1,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--out", default="runtime/gsvq_iquant_sweep")
    parser.add_argument("--preset", default="tuned", choices=sorted(PRESETS))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-bits", type=float, default=4.0)
    parser.add_argument("--layers", default="", help="Comma-separated layer indices to include.")
    parser.add_argument("--qtypes", default="", help="Comma-separated qtypes to include.")
    parser.add_argument("--max-tensors", type=int, default=0)
    parser.add_argument("--chunk-vectors", type=int, default=65536)
    parser.add_argument("--max-vectors-per-tensor", type=int, default=0,
                        help="0 means full tensor; useful for hparam probes.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--temp-start", type=float, default=None)
    parser.add_argument("--temp-end", type=float, default=None)
    parser.add_argument("--scale-start", type=float, default=None)
    parser.add_argument("--scale-end", type=float, default=None)
    parser.add_argument("--candidate-count", type=int, default=None)
    parser.add_argument("--neighbor-candidates", type=int, default=None)
    parser.add_argument("--target-candidates", type=int, default=None)
    parser.add_argument("--rotation-trick", action="store_true")
    parser.add_argument("--no-acceptance-guard", action="store_true")
    parser.add_argument("--std", type=float, default=None)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--optimizer-name", default=None, choices=["adamw", "adam", "sgd", "lion"])
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--restarts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _params(args):
    params = dict(PRESETS[args.preset])
    for key in (
        "steps",
        "lr",
        "weight_decay",
        "candidate_count",
        "neighbor_candidates",
        "target_candidates",
        "std",
        "strength",
        "optimizer_name",
        "grad_clip",
        "restarts",
    ):
        value = getattr(args, key.replace("-", "_"), None)
        if value is not None:
            params[key] = value
    if args.temp_start is not None or args.temp_end is not None:
        params["temperature"] = (
            args.temp_start if args.temp_start is not None else params["temperature"][0],
            args.temp_end if args.temp_end is not None else params["temperature"][1],
        )
    if args.scale_start is not None or args.scale_end is not None:
        params["logit_scale"] = (
            args.scale_start if args.scale_start is not None else params["logit_scale"][0],
            args.scale_end if args.scale_end is not None else params["logit_scale"][1],
        )
    return params


def _load_hf_tensor(model_dir: str | Path, tensor_name: str) -> torch.Tensor:
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r") as f:
            weight_map = json.load(f)["weight_map"]
        if tensor_name not in weight_map:
            raise KeyError(f"{tensor_name!r} not found in {index_path}")
        shard = model_dir / weight_map[tensor_name]
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            return f.get_tensor(tensor_name).float()
    single = model_dir / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt", device="cpu") as f:
            return f.get_tensor(tensor_name).float()
    raise FileNotFoundError(f"No safetensors model found under {model_dir}")


def _layer_idx(name: str) -> int | None:
    m = re.search(r"(?:blk\.|layers\.)(\d+)", name)
    return int(m.group(1)) if m else None


def _parse_int_set(raw: str) -> set[int] | None:
    raw = raw.strip()
    if not raw:
        return None
    return {int(x) for x in raw.split(",") if x.strip()}


def _parse_str_set(raw: str) -> set[str] | None:
    raw = raw.strip()
    if not raw:
        return None
    return {x.strip().upper() for x in raw.split(",") if x.strip()}


def _make_quantizer(decomp, target_vectors, start, end, params, device, rotation_trick):
    if hasattr(decomp, "half_magnitude_codebook"):
        return PairedMagnitudeIQuantGSVQ(
            target_vectors,
            decomp.vector_scales[start:end].to(device),
            decomp.half_magnitude_codebook.to(device),
            decomp.sign_codebook.to(device),
            decomp.first_magnitude_indices[start:end].to(device),
            decomp.second_magnitude_indices[start:end].to(device),
            decomp.sign_indices[start:end].to(device),
            candidate_count=params["candidate_count"],
            neighbor_candidates=params["neighbor_candidates"],
            target_candidates=params["target_candidates"],
            std=params["std"],
            strength=params["strength"],
            rotation_trick=rotation_trick,
        )
    return FactorizedIQuantGSVQ(
        target_vectors,
        decomp.vector_scales[start:end].to(device),
        decomp.magnitude_codebook.to(device),
        decomp.sign_codebook.to(device),
        decomp.magnitude_indices[start:end].to(device),
        decomp.sign_indices[start:end].to(device),
        candidate_count=params["candidate_count"],
        neighbor_candidates=params["neighbor_candidates"],
        target_candidates=params["target_candidates"],
        std=params["std"],
        strength=params["strength"],
        rotation_trick=rotation_trick,
    )


def _initial_vectors(decomp, start, end, device):
    if hasattr(decomp, "half_magnitude_codebook"):
        mag = torch.cat([
            decomp.half_magnitude_codebook[decomp.first_magnitude_indices[start:end]],
            decomp.half_magnitude_codebook[decomp.second_magnitude_indices[start:end]],
        ], dim=-1)
        return (
            decomp.vector_scales[start:end]
            * mag
            * decomp.sign_codebook[decomp.sign_indices[start:end]]
        ).to(device)
    return (
        decomp.vector_scales[start:end]
        * decomp.magnitude_codebook[decomp.magnitude_indices[start:end]]
        * decomp.sign_codebook[decomp.sign_indices[start:end]]
    ).to(device)


def run_tensor(store, ref, args, params):
    device = torch.device(args.device)
    decomp = store.decompose(ref.gguf_name, device="cpu")
    hf_module = decomp.hf_name or gguf_name_to_hf_name(decomp.gguf_name)
    if hf_module is None:
        raise ValueError(f"Cannot map {ref.gguf_name} to HF tensor")
    hf_name = hf_module + ".weight"
    target = _load_hf_tensor(args.hf_model, hf_name)
    if tuple(target.shape) != decomp.original_shape:
        raise ValueError(f"{hf_name}: HF shape {tuple(target.shape)} != GGUF shape {decomp.original_shape}")

    vector_dim = decomp.vector_dim
    target_vectors_cpu = target.reshape(-1, vector_dim)
    total_vectors = target_vectors_cpu.shape[0]
    if args.max_vectors_per_tensor > 0:
        total_vectors = min(total_vectors, args.max_vectors_per_tensor)

    chunk = min(args.chunk_vectors, total_vectors)
    initial_sse = 0.0
    raw_best_sse = 0.0
    accepted_sse = 0.0
    improved_chunks = 0
    chunks = 0
    start_time = time.time()

    for start in range(0, total_vectors, chunk):
        end = min(start + chunk, total_vectors)
        target_vectors = target_vectors_cpu[start:end].to(device)
        init_vectors = _initial_vectors(decomp, start, end, device)
        init_mse = (init_vectors - target_vectors).square().mean().item()
        best_history = None
        best_quantizer = None
        for restart in range(max(1, int(params["restarts"]))):
            torch.manual_seed(args.seed + 1009 * restart + start)
            quantizer = _make_quantizer(decomp, target_vectors, start, end, params, device, args.rotation_trick)
            history = train_gsvq_reconstruction(
                quantizer,
                steps=params["steps"],
                lr=params["lr"],
                temperature=params["temperature"],
                logit_scale=params["logit_scale"],
                weight_decay=params["weight_decay"],
                optimizer_name=params["optimizer_name"],
                grad_clip=params["grad_clip"],
            )
            if best_history is None or history.best_hard_mse < best_history.best_hard_mse:
                del best_quantizer
                best_history = history
                best_quantizer = quantizer
            else:
                del quantizer
        history = best_history
        elems = (end - start) * vector_dim
        initial_sse += init_mse * elems
        raw_best_sse += history.best_hard_mse * elems
        accepted_mse = history.best_hard_mse
        if history.best_hard_mse < init_mse:
            improved_chunks += 1
        elif not args.no_acceptance_guard:
            accepted_mse = init_mse
        accepted_sse += accepted_mse * elems
        chunks += 1

        del best_quantizer, target_vectors, init_vectors
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()

    elems_total = total_vectors * vector_dim
    init = initial_sse / elems_total
    raw = raw_best_sse / elems_total
    accepted = accepted_sse / elems_total
    return {
        "gguf_tensor": ref.gguf_name,
        "hf_tensor": hf_name,
        "layer": _layer_idx(ref.gguf_name),
        "qtype": ref.qtype_name,
        "bits_per_weight": ref.bits_per_weight,
        "shape": list(decomp.original_shape),
        "vectors": int(total_vectors),
        "vector_dim": int(vector_dim),
        "chunks": chunks,
        "improved_chunks": improved_chunks,
        "initial_mse": init,
        "raw_best_mse": raw,
        "accepted_mse": accepted,
        "raw_abs_delta": init - raw,
        "accepted_abs_delta": init - accepted,
        "raw_rel_delta": (init - raw) / max(init, 1e-30),
        "accepted_rel_delta": (init - accepted) / max(init, 1e-30),
        "elapsed_sec": time.time() - start_time,
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_layers(rows):
    agg = {}
    for row in rows:
        layer = row["layer"]
        elems = row["vectors"] * row["vector_dim"]
        if layer not in agg:
            agg[layer] = {
                "layer": layer,
                "tensors": 0,
                "elements": 0,
                "initial_sse": 0.0,
                "raw_best_sse": 0.0,
                "accepted_sse": 0.0,
                "elapsed_sec": 0.0,
            }
        a = agg[layer]
        a["tensors"] += 1
        a["elements"] += elems
        a["initial_sse"] += row["initial_mse"] * elems
        a["raw_best_sse"] += row["raw_best_mse"] * elems
        a["accepted_sse"] += row["accepted_mse"] * elems
        a["elapsed_sec"] += row["elapsed_sec"]
    out = []
    for a in sorted(agg.values(), key=lambda x: (x["layer"] is None, x["layer"])):
        initial = a["initial_sse"] / a["elements"]
        raw = a["raw_best_sse"] / a["elements"]
        accepted = a["accepted_sse"] / a["elements"]
        out.append({
            "layer": a["layer"],
            "tensors": a["tensors"],
            "elements": a["elements"],
            "initial_mse": initial,
            "raw_best_mse": raw,
            "accepted_mse": accepted,
            "raw_abs_delta": initial - raw,
            "accepted_abs_delta": initial - accepted,
            "raw_rel_delta": (initial - raw) / max(initial, 1e-30),
            "accepted_rel_delta": (initial - accepted) / max(initial, 1e-30),
            "elapsed_sec": a["elapsed_sec"],
        })
    return out


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    params = _params(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = GGUFIQuantStore(args.gguf)
    refs = list(store.collect_refs(max_bits=args.max_bits).values())
    layers = _parse_int_set(args.layers)
    qtypes = _parse_str_set(args.qtypes)
    if layers is not None:
        refs = [r for r in refs if _layer_idx(r.gguf_name) in layers]
    if qtypes is not None:
        refs = [r for r in refs if r.qtype_name.upper() in qtypes]
    refs = [r for r in refs if r.hf_name is not None]
    refs.sort(key=lambda r: (_layer_idx(r.gguf_name) is None, _layer_idx(r.gguf_name) or -1, r.gguf_name))
    if args.max_tensors > 0:
        refs = refs[:args.max_tensors]

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = out_dir / f"{run_id}_{args.preset}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump({"args": vars(args), "params": params, "num_tensors": len(refs)}, f, indent=2, sort_keys=True)

    rows = []
    print(f"GSVQ IQuant sweep: tensors={len(refs)} preset={args.preset} device={args.device}", flush=True)
    for idx, ref in enumerate(refs, start=1):
        print(f"[{idx}/{len(refs)}] {ref.gguf_name} {ref.qtype_name} shape={ref.shape}", flush=True)
        row = run_tensor(store, ref, args, params)
        rows.append(row)
        with open(run_dir / "tensor_results.jsonl", "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"  init={row['initial_mse']:.8e} accepted={row['accepted_mse']:.8e} "
            f"delta={row['accepted_abs_delta']:.8e} rel={100 * row['accepted_rel_delta']:.3f}% "
            f"chunks={row['improved_chunks']}/{row['chunks']} time={row['elapsed_sec']:.1f}s",
            flush=True,
        )
        gc.collect()

    layer_rows = aggregate_layers(rows)
    write_csv(run_dir / "tensor_results.csv", rows)
    write_csv(run_dir / "layer_results.csv", layer_rows)
    with open(run_dir / "layer_results.json", "w") as f:
        json.dump(layer_rows, f, indent=2, sort_keys=True)

    print("\nPer-layer accepted reconstruction MSE reduction:", flush=True)
    for row in layer_rows:
        print(
            f"layer={row['layer']} tensors={row['tensors']} "
            f"{row['initial_mse']:.8e}->{row['accepted_mse']:.8e} "
            f"delta={row['accepted_abs_delta']:.8e} rel={100 * row['accepted_rel_delta']:.3f}%",
            flush=True,
        )
    print(f"\nWrote results to {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
