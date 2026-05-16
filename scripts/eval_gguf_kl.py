#!/usr/bin/env python3
"""Evaluate KL(ref dense HF || GGUF-dequantized HF model) for two GGUF files."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gguf_iq import GGUFIQuantStore, gguf_name_to_hf_name  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--baseline-gguf", required=True)
    parser.add_argument("--candidate-gguf", required=True)
    parser.add_argument("--out", default="runtime/gsvq_iquant_kl")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--num-sequences", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _hf_weight_name(gguf_name: str) -> str | None:
    module_name = gguf_name_to_hf_name(gguf_name)
    if module_name is None:
        return None
    return module_name + ".weight"


def load_model(path: str, *, device: torch.device, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    return model


def load_gguf_weights_into_model(model, gguf_path: str, *, device: torch.device) -> dict:
    store = GGUFIQuantStore(gguf_path)
    params = dict(model.named_parameters())
    loaded = 0
    skipped = []
    started = time.time()

    for idx, tensor in enumerate(store.reader.tensors, start=1):
        param_name = _hf_weight_name(tensor.name)
        if param_name is None or param_name not in params:
            skipped.append(tensor.name)
            continue
        param = params[param_name]
        dense = store.dequantize_dense(tensor.name, device="cpu", dtype=torch.float32)
        if tuple(dense.shape) != tuple(param.shape):
            raise ValueError(
                f"{tensor.name} -> {param_name}: GGUF dense shape {tuple(dense.shape)} "
                f"does not match HF shape {tuple(param.shape)}"
            )
        with torch.no_grad():
            param.copy_(dense.to(device=device, dtype=param.dtype, non_blocking=True))
        loaded += 1
        del dense
        if device.type == "cuda" and idx % 25 == 0:
            torch.cuda.empty_cache()
        if idx % 50 == 0:
            print(f"  loaded {idx}/{len(store.reader.tensors)} GGUF tensors", flush=True)

    model.tie_weights()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "loaded_tensors": loaded,
        "skipped_tensors": skipped,
        "elapsed_sec": time.time() - started,
    }


def build_eval_batches(args, tokenizer) -> list[torch.Tensor]:
    from datasets import load_dataset

    try:
        dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
        texts = [row[args.text_field] for row in dataset if row.get(args.text_field, "").strip()]
        source = "\n\n".join(texts)
    except Exception as exc:
        print(f"Dataset load failed ({exc}); using built-in fallback text.", flush=True)
        source = (
            "Large language models are evaluated by comparing probability distributions "
            "over the next token on held-out text. This fallback is only for smoke tests.\n"
        ) * 4096

    encoded = tokenizer(source, return_tensors="pt", add_special_tokens=False).input_ids[0]
    needed = args.num_sequences * args.seq_len
    if encoded.numel() < needed + 1:
        repeats = (needed + 1 + encoded.numel() - 1) // max(1, encoded.numel())
        encoded = encoded.repeat(repeats)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    max_start = encoded.numel() - args.seq_len - 1
    starts = torch.linspace(0, max_start, steps=args.num_sequences).round().long()
    if args.num_sequences > 1:
        jitter = torch.randint(0, max(1, args.seq_len // 2), (args.num_sequences,), generator=generator)
        starts = (starts + jitter).clamp(max=max_start)
    sequences = [encoded[start:start + args.seq_len].clone() for start in starts.tolist()]
    batches = []
    for start in range(0, len(sequences), args.batch_size):
        batches.append(torch.stack(sequences[start:start + args.batch_size], dim=0))
    return batches


@torch.inference_mode()
def evaluate_kl(ref_model, eval_model, batches: list[torch.Tensor], *, device: torch.device) -> dict:
    total_kl = 0.0
    total_tokens = 0
    started = time.time()
    for idx, input_ids in enumerate(batches, start=1):
        input_ids = input_ids.to(device)
        ref_logits = ref_model(input_ids=input_ids).logits[:, :-1, :].float()
        eval_logits = eval_model(input_ids=input_ids).logits[:, :-1, :].float()
        ref_logp = F.log_softmax(ref_logits, dim=-1)
        eval_logp = F.log_softmax(eval_logits, dim=-1)
        kl = (ref_logp.exp() * (ref_logp - eval_logp)).sum(dim=-1)
        total_kl += kl.sum().item()
        total_tokens += kl.numel()
        del input_ids, ref_logits, eval_logits, ref_logp, eval_logp, kl
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if idx % 8 == 0 or idx == len(batches):
            print(f"  evaluated {idx}/{len(batches)} batches", flush=True)
    return {
        "kl_nats_per_token": total_kl / max(1, total_tokens),
        "tokens": total_tokens,
        "elapsed_sec": time.time() - started,
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out) / time.strftime("%Y%m%d-%H%M%S_kl")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = _torch_dtype(args.dtype)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    batches = build_eval_batches(args, tokenizer)
    total_tokens = sum(batch.numel() - batch.shape[0] for batch in batches)
    print(
        f"KL eval data: {len(batches)} batches, {total_tokens} next-token positions, "
        f"seq_len={args.seq_len}, dataset={args.dataset_name}/{args.dataset_config}:{args.dataset_split}",
        flush=True,
    )

    print("Loading dense HF reference model", flush=True)
    ref_model = load_model(args.hf_model, device=device, dtype=dtype)
    print("Loading reusable HF model shell for GGUF weights", flush=True)
    eval_model = load_model(args.hf_model, device=device, dtype=dtype)

    results = {"args": vars(args), "data_tokens": total_tokens}
    for label, gguf_path in (("baseline", args.baseline_gguf), ("candidate", args.candidate_gguf)):
        print(f"\nLoading {label} GGUF weights: {gguf_path}", flush=True)
        load_info = load_gguf_weights_into_model(eval_model, gguf_path, device=device)
        print(
            f"Loaded {load_info['loaded_tensors']} tensors in {load_info['elapsed_sec']:.1f}s; "
            f"skipped={len(load_info['skipped_tensors'])}",
            flush=True,
        )
        print(f"Evaluating {label} KL", flush=True)
        metrics = evaluate_kl(ref_model, eval_model, batches, device=device)
        results[label] = {"gguf": gguf_path, "load": load_info, "metrics": metrics}
        print(
            f"{label}: KL={metrics['kl_nats_per_token']:.8e} nats/token "
            f"tokens={metrics['tokens']} time={metrics['elapsed_sec']:.1f}s",
            flush=True,
        )
        gc.collect()

    base_kl = results["baseline"]["metrics"]["kl_nats_per_token"]
    cand_kl = results["candidate"]["metrics"]["kl_nats_per_token"]
    results["delta"] = {
        "kl_abs_delta": base_kl - cand_kl,
        "kl_rel_delta": (base_kl - cand_kl) / max(base_kl, 1e-30),
    }
    with open(out_dir / "kl_results.json", "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(
        f"\nKL delta baseline-candidate={results['delta']['kl_abs_delta']:.8e} "
        f"rel={100 * results['delta']['kl_rel_delta']:.4f}%",
        flush=True,
    )
    print(f"Wrote KL results to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
