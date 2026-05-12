"""Assemble quantized per-expert checkpoint files into a HuggingFace-compatible model.

Usage:
    python save_model.py --config configs/local/config.yaml                     # uses latest run
    python save_model.py --config configs/local/config.yaml --run-id <run_id>   # specific run
    python save_model.py --config configs/local/config.yaml --out-dir ./output  # custom output dir
"""
import os
import re
import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from safetensors.torch import safe_open, save_file
from transformers import AutoConfig
from src.config import load_config


def resolve_model_path(model_name):
    p = Path(model_name)
    if p.exists():
        return str(p.parent.resolve()) if p.is_file() else str(p.resolve())
    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(repo_id=model_name)
        return str(Path(local_dir).resolve())
    except Exception as e:
        raise FileNotFoundError(
            f"Could not resolve '{model_name}' as a local path or HF Hub repo. "
            f"Original error: {e}"
        )


def find_latest_run(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = []
    for name in os.listdir(checkpoint_dir):
        run_path = os.path.join(checkpoint_dir, name)
        prog_path = os.path.join(run_path, "progress.json")
        if os.path.isdir(run_path) and os.path.exists(prog_path):
            mtime = os.path.getmtime(prog_path)
            candidates.append((mtime, name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def collect_base_model_entries(base_model_dir):
    """Collect ALL tensors from the original base model."""
    entries = {}
    for fname in sorted(os.listdir(base_model_dir)):
        if not fname.endswith(".safetensors"):
            continue
        path = os.path.join(base_model_dir, fname)
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                m = re.search(r"\.layers\.(\d+)\.", k)
                layer_idx = int(m.group(1)) if m else None
                t = f.get_tensor(k)
                entries[k] = {
                    "file": path,
                    "key": k,
                    "nbytes": t.numel() * t.element_size(),
                    "layer_idx": layer_idx,
                }
    return entries


def collect_quantized_entries(per_expert_dir):
    """Collect quantized tensors from the run checkpoint directory.

    Picks up all .safetensors files (experts, shared_experts, etc.) except
    progress.json and other non-tensor files. Each file's layer index is
    extracted from its name.
    """
    entries = {}
    for fname in sorted(os.listdir(per_expert_dir)):
        if not fname.endswith(".safetensors"):
            continue
        m = re.search(r"layers_(\d+)", fname)
        if m is None:
            continue
        layer_idx = int(m.group(1))
        path = os.path.join(per_expert_dir, fname)
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                entries[k] = {
                    "file": path,
                    "key": k,
                    "nbytes": t.numel() * t.element_size(),
                    "layer_idx": layer_idx,
                }
    return entries


def copy_non_weight_files(base_model_dir, out_dir):
    """Copy config, tokenizer, and other metadata files from the base model."""
    patterns = [
        "config.json", "generation_config.json",
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.json", "merges.txt",
        "tokenizer.model", "tiktoken.model",
        "preprocessor_config.json",
    ]
    copied = []
    for fname in os.listdir(base_model_dir):
        if fname in patterns or fname.endswith(".py"):
            src = os.path.join(base_model_dir, fname)
            dst = os.path.join(out_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                copied.append(fname)
    return copied


def _build_ignore_list(model_config):
    """Build the quantization ignore list based on model architecture.

    Only MLP expert projections (gate_proj, up_proj, down_proj) are quantized.
    Everything else -- attention, norms, embeddings, routing gates, shared
    experts, vision towers -- must be excluded.
    """
    ignore = [
        "lm_head",
        "re:.*embed_tokens.*",
        "re:.*self_attn.*",
        "re:.*input_layernorm.*",
        "re:.*post_attention_layernorm.*",
        r"re:.*\.norm$",
    ]

    text_config = model_config.get("text_config", {})
    model_type = model_config.get("model_type", "")

    if "vision_config" in model_config or "vision_tower" in model_type:
        ignore.append("re:.*vision_tower.*")
        ignore.append("re:.*mm_projector.*")

    merged = {**model_config, **text_config}

    has_linear_attn = "layer_types" in merged
    if has_linear_attn:
        ignore.append("re:.*linear_attn.*")

    is_moe = merged.get("num_experts", 0) > 0 or merged.get("n_routed_experts", 0) > 0
    if is_moe:
        ignore.append(r"re:.*mlp\.gate$")

    has_shared_expert = merged.get("n_shared_experts", 0) > 0 or "qwen3_5" in model_type.lower()
    if has_shared_expert:
        ignore.append("re:.*shared_expert.*")

    first_k_dense = merged.get("first_k_dense_replace", 0)
    if first_k_dense > 0:
        dense_indices = "|".join(str(i) for i in range(first_k_dense))
        ignore.append(f"re:.*layers\\.({dense_indices})\\.mlp\\.(gate_proj|up_proj|down_proj|gate_up_proj).*")

    return ignore


def inject_quantization_config(out_dir, groupsize, wbits=4):
    """Add compressed-tensors quantization_config to config.json.

    This enables vLLM and HuggingFace transformers to auto-detect the
    pack-quantized format when loading the assembled model.
    """
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    ignore = _build_ignore_list(config)

    config["quantization_config"] = {
        "config_groups": {
            "group_0": {
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "num_bits": wbits,
                    "strategy": "group",
                    "group_size": groupsize,
                    "symmetric": True,
                    "type": "int",
                },
            }
        },
        "format": "pack-quantized",
        "ignore": ignore,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }

    if "text_config" in config and "quantization_config" in config["text_config"]:
        del config["text_config"]["quantization_config"]

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Assemble quantized expert weights into a HF-compatible model checkpoint")
    parser.add_argument("--config", type=str, default="configs/local/config.yaml",
                        help="Path to the training config YAML")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID to export. Defaults to the latest completed run.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory. Defaults to <checkpoint_dir>/<run_id>/assembled")
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_name = cfg.model.name
    checkpoint_dir = cfg.training.checkpoint_dir

    if args.run_id is not None:
        run_id = args.run_id
    else:
        run_id = find_latest_run(checkpoint_dir)
        if run_id is None:
            raise RuntimeError(
                f"No completed runs found in '{checkpoint_dir}'. "
                "Pass --run-id explicitly or complete a training run first.")

    run_dir = os.path.join(checkpoint_dir, run_id)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    out_dir = args.out_dir or os.path.join(run_dir, "assembled")

    print(f"Model:        {model_name}")
    print(f"Run ID:       {run_id}")
    print(f"Run dir:      {run_dir}")
    print(f"Output dir:   {out_dir}")
    print()

    base_model_dir = resolve_model_path(model_name)
    cfg_hf = AutoConfig.from_pretrained(base_model_dir, trust_remote_code=True)
    num_layers = getattr(cfg_hf, "num_hidden_layers", None)
    if num_layers is None and hasattr(cfg_hf, "text_config"):
        num_layers = cfg_hf.text_config.num_hidden_layers
    if num_layers is None:
        raise RuntimeError(
            f"Cannot determine num_hidden_layers from {base_model_dir}/config.json. "
            "Check model config structure.")
    num_shards = num_layers + 1

    print(f"Base model:   {base_model_dir}")
    print(f"Layers:       {num_layers}")
    print(f"Shards:       {num_shards}")
    print()

    print("Collecting base model tensors...")
    base_entries = collect_base_model_entries(base_model_dir)
    print(f"  Found {len(base_entries)} tensors")

    print("Collecting quantized tensors...")
    quantized_entries = collect_quantized_entries(run_dir)
    print(f"  Found {len(quantized_entries)} tensors")

    if not quantized_entries:
        raise RuntimeError(
            f"No quantized shard files found in {run_dir}. "
            "Make sure the training run completed at least one layer.")

    overridden = set(base_entries.keys()) & set(quantized_entries.keys())
    print(f"  Overriding {len(overridden)} base tensors with quantized versions")

    merged = {**base_entries, **quantized_entries}
    all_entries = list(merged.values())

    for e in all_entries:
        if e["layer_idx"] is None:
            e["shard_idx"] = 0
        else:
            e["shard_idx"] = e["layer_idx"] + 1

    shard_sizes = [0] * num_shards
    for e in all_entries:
        shard_sizes[e["shard_idx"]] += e["nbytes"]

    print("\nShard sizes:")
    for idx, size in enumerate(shard_sizes):
        if size > 0:
            print(f"  shard {idx:03d}: {size / (1024**3):.3f} GB")

    os.makedirs(out_dir, exist_ok=True)

    def make_shard_name(idx):
        return f"model-{idx+1:05d}-of-{num_shards:05d}.safetensors"

    entries_by_file = defaultdict(list)
    for e in all_entries:
        entries_by_file[e["file"]].append(e)

    shard_tensors = [dict() for _ in range(num_shards)]
    weight_map = {}
    total_size = 0

    print("\nLoading tensors...")
    for src_file, file_entries in entries_by_file.items():
        with safe_open(src_file, framework="pt", device="cpu") as f:
            for e in file_entries:
                t = f.get_tensor(e["key"])
                shard_idx = e["shard_idx"]
                shard_name = make_shard_name(shard_idx)
                shard_tensors[shard_idx][e["key"]] = t
                weight_map[e["key"]] = shard_name
                total_size += e["nbytes"]

    print(f"\nWriting {num_shards} shards to {out_dir}...")
    for shard_idx in range(num_shards):
        if not shard_tensors[shard_idx]:
            continue
        shard_name = make_shard_name(shard_idx)
        shard_path = os.path.join(out_dir, shard_name)
        print(f"  {shard_name}: {len(shard_tensors[shard_idx])} tensors")
        save_file(shard_tensors[shard_idx], shard_path)

    index = {
        "metadata": {"total_size": int(total_size)},
        "weight_map": weight_map,
    }
    index_path = os.path.join(out_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print("\nCopying config & tokenizer files...")
    copied = copy_non_weight_files(base_model_dir, out_dir)
    for fname in copied:
        print(f"  {fname}")

    groupsize = cfg.quantization.groupsize
    print(f"\nInjecting quantization_config into config.json (group_size={groupsize})...")
    inject_quantization_config(out_dir, groupsize)

    print(f"\nDone. Total model size: ~{total_size / (1024**3):.2f} GB")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
