#!/usr/bin/env python3
"""Patch a copied GGUF with fixed-scale GSVQ-optimized IQuant assignments."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_gsvq_iquant_sweep import (  # noqa: E402
    PRESETS,
    _initial_vectors,
    _layer_idx,
    _load_hf_tensor,
    _make_quantizer,
    _params,
    _parse_int_set,
    _parse_str_set,
    aggregate_layers,
    write_csv,
)
from src.gguf_iq import GGUFIQuantStore, gguf_name_to_hf_name  # noqa: E402
from src.quantization.gsvq import train_gsvq_reconstruction  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True, help="Input Unsloth GGUF.")
    parser.add_argument("--hf-model", required=True, help="Dense HF safetensors directory used as the target.")
    parser.add_argument("--out-gguf", required=True, help="Output GGUF path to create and patch.")
    parser.add_argument("--out", default="runtime/gsvq_iquant_requantized")
    parser.add_argument("--preset", default="tuned", choices=sorted(PRESETS))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-bits", type=float, default=4.0)
    parser.add_argument("--layers", default="", help="Comma-separated layer indices to include.")
    parser.add_argument("--qtypes", default="", help="Comma-separated qtypes to include.")
    parser.add_argument("--max-tensors", type=int, default=0)
    parser.add_argument("--chunk-vectors", type=int, default=65536)
    parser.add_argument("--max-vectors-per-tensor", type=int, default=0,
                        help="0 means full tensor; useful only for smoke tests.")
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
    parser.add_argument("--init-mode", default=None,
                        choices=["binary", "prior", "target", "posterior",
                                 "prior_delta", "target_delta", "posterior_delta"])
    parser.add_argument("--prior-weight", type=float, default=None)
    parser.add_argument("--target-weight", type=float, default=None)
    parser.add_argument("--prior-radius-k", type=int, default=None)
    parser.add_argument("--prior-radius-scale", type=float, default=None)
    parser.add_argument("--target-norm-scale", type=float, default=None)
    parser.add_argument("--posterior-current-bias", type=float, default=None)
    parser.add_argument("--joint-init", action="store_true")
    parser.add_argument("--joint-init-max-options", type=int, default=None)
    parser.add_argument("--optimizer-name", default=None, choices=["adamw", "adam", "sgd", "lion"])
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--restarts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-patch", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _prepare_output(input_path: Path, output_path: Path, overwrite: bool) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--out-gguf must be a separate path; in-place patching is intentionally disabled")
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)


def _init_assignment_buffers(decomp):
    if hasattr(decomp, "half_magnitude_codebook"):
        return {
            "first_magnitude_indices": decomp.first_magnitude_indices.detach().cpu().numpy().copy(),
            "second_magnitude_indices": decomp.second_magnitude_indices.detach().cpu().numpy().copy(),
            "sign_indices": decomp.sign_indices.detach().cpu().numpy().copy(),
        }
    return {
        "magnitude_indices": decomp.magnitude_indices.detach().cpu().numpy().copy(),
        "sign_indices": decomp.sign_indices.detach().cpu().numpy().copy(),
    }


def _store_hard_indices(buffers, start: int, end: int, quantizer) -> None:
    indices = quantizer.get_hard_indices()
    for key, value in indices.items():
        buffers[key][start:end] = value.detach().cpu().numpy()


def _patch_tensor(store: GGUFIQuantStore, name: str, buffers) -> None:
    if "magnitude_indices" in buffers:
        store.patch_tensor_indices(
            name,
            magnitude_indices=buffers["magnitude_indices"],
            sign_indices=buffers["sign_indices"],
        )
    else:
        store.patch_tensor_indices(
            name,
            first_magnitude_indices=buffers["first_magnitude_indices"],
            second_magnitude_indices=buffers["second_magnitude_indices"],
            sign_indices=buffers["sign_indices"],
        )


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
    partial_tensor = False
    if args.max_vectors_per_tensor > 0:
        total_vectors = min(total_vectors, args.max_vectors_per_tensor)
        partial_tensor = total_vectors < target_vectors_cpu.shape[0]

    buffers = _init_assignment_buffers(decomp)
    chunk = min(args.chunk_vectors, total_vectors)
    initial_sse = 0.0
    raw_best_sse = 0.0
    accepted_sse = 0.0
    improved_chunks = 0
    patched_chunks = 0
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
        accepted_mse = init_mse
        accept = history.best_hard_mse < init_mse or args.no_acceptance_guard
        if history.best_hard_mse < init_mse:
            improved_chunks += 1
        if accept:
            _store_hard_indices(buffers, start, end, best_quantizer)
            accepted_mse = history.best_hard_mse
            patched_chunks += 1
        accepted_sse += accepted_mse * elems
        chunks += 1

        del best_quantizer, target_vectors, init_vectors
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()

    _patch_tensor(store, ref.gguf_name, buffers)
    store.flush()

    elems_total = total_vectors * vector_dim
    init = initial_sse / elems_total
    raw = raw_best_sse / elems_total
    accepted = accepted_sse / elems_total
    patch_mse = None
    if args.verify_patch and not partial_tensor:
        patched = store.dequantize_dense(ref.gguf_name, device="cpu", dtype=torch.float32)
        patch_mse = (patched - target).square().mean().item()
        del patched

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
        "patched_chunks": patched_chunks,
        "initial_mse": init,
        "raw_best_mse": raw,
        "accepted_mse": accepted,
        "patched_mse": patch_mse,
        "raw_abs_delta": init - raw,
        "accepted_abs_delta": init - accepted,
        "raw_rel_delta": (init - raw) / max(init, 1e-30),
        "accepted_rel_delta": (init - accepted) / max(init, 1e-30),
        "elapsed_sec": time.time() - start_time,
    }


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    params = _params(args)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    input_gguf = Path(args.gguf).expanduser()
    output_gguf = Path(args.out_gguf).expanduser()
    print(f"Copying {input_gguf} -> {output_gguf}", flush=True)
    _prepare_output(input_gguf, output_gguf, args.overwrite)

    store = GGUFIQuantStore(output_gguf, mode="r+")
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
    run_dir = out_root / f"{run_id}_{args.preset}_requant"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "params": params,
                "num_tensors": len(refs),
                "input_gguf": str(input_gguf),
                "output_gguf": str(output_gguf),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    rows = []
    print(f"GSVQ IQuant GGUF patch: tensors={len(refs)} preset={args.preset} device={args.device}", flush=True)
    for idx, ref in enumerate(refs, start=1):
        print(f"[{idx}/{len(refs)}] {ref.gguf_name} {ref.qtype_name} shape={ref.shape}", flush=True)
        row = run_tensor(store, ref, args, params)
        rows.append(row)
        with open(run_dir / "tensor_results.jsonl", "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        patch_note = "" if row["patched_mse"] is None else f" patched={row['patched_mse']:.8e}"
        print(
            f"  init={row['initial_mse']:.8e} accepted={row['accepted_mse']:.8e}{patch_note} "
            f"delta={row['accepted_abs_delta']:.8e} rel={100 * row['accepted_rel_delta']:.3f}% "
            f"patched_chunks={row['patched_chunks']}/{row['chunks']} time={row['elapsed_sec']:.1f}s",
            flush=True,
        )
        gc.collect()

    store.flush()
    layer_rows = aggregate_layers(rows)
    write_csv(run_dir / "tensor_results.csv", rows)
    write_csv(run_dir / "layer_results.csv", layer_rows)
    with open(run_dir / "layer_results.json", "w") as f:
        json.dump(layer_rows, f, indent=2, sort_keys=True)

    total_elems = sum(row["vectors"] * row["vector_dim"] for row in rows)
    total_init = sum(row["initial_mse"] * row["vectors"] * row["vector_dim"] for row in rows) / total_elems
    total_acc = sum(row["accepted_mse"] * row["vectors"] * row["vector_dim"] for row in rows) / total_elems
    print("\nPer-layer accepted reconstruction MSE reduction:", flush=True)
    for row in layer_rows:
        print(
            f"layer={row['layer']} tensors={row['tensors']} "
            f"{row['initial_mse']:.8e}->{row['accepted_mse']:.8e} "
            f"delta={row['accepted_abs_delta']:.8e} rel={100 * row['accepted_rel_delta']:.3f}%",
            flush=True,
        )
    print(
        f"\nAll selected IQuant tensors: {total_init:.12e}->{total_acc:.12e} "
        f"rel={100 * (total_init - total_acc) / max(total_init, 1e-30):.4f}%",
        flush=True,
    )
    print(f"Wrote patched GGUF to {output_gguf}", flush=True)
    print(f"Wrote results to {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
