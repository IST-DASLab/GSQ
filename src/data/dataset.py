import os
import glob
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from torch.utils.data import DataLoader
from datasets import load_dataset
import random
import re

def split_thought_solution(text: str):
    thought_re = re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.DOTALL)
    solution_re = re.compile(r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", re.DOTALL)

    thought = thought_re.search(text).group(1).strip()
    solution = solution_re.search(text).group(1).strip()

    return thought, solution

def _get_dist_info():
    import os
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size

def _shard_list_contiguous(xs, rank, world_size):
    n = len(xs)
    start = (n * rank) // world_size
    end = (n * (rank + 1)) // world_size
    return xs[start:end]

def _maybe_shard(xs, dont_shard=False):
    rank, world_size = _get_dist_info()
    if world_size <= 1 or dont_shard:
        return xs
    return _shard_list_contiguous(xs, rank, world_size)

def make_concat_chunks(data, tokenizer, max_length, num_samples, seed=0, add_eos=False):
    eos_id = tokenizer.eos_token_id
    token_buffer = []
    chunks = []
    for ex in data:
        ids = tokenizer(ex["text"], return_tensors=None)["input_ids"]
        if add_eos and eos_id is not None:
            ids = ids + [eos_id]
        token_buffer.extend(ids)

        while len(token_buffer) >= max_length:
            chunk = token_buffer[:max_length]
            del token_buffer[:max_length]
            chunks.append(torch.tensor(chunk, dtype=torch.long))
            if len(chunks) >= num_samples:
                return chunks

    return chunks

def tokenize_texts(data, tokenizer, max_length, num_samples, seed=0):
    random.seed(seed)
    tokenized_batches = []

    for i, text in enumerate(data):
        if len(tokenized_batches) >= num_samples:
            break
        enc = tokenizer(text["text"], return_tensors="pt")
        if enc.input_ids.shape[1] >= max_length + 1:
            start = random.randint(0, enc.input_ids.shape[1] - max_length - 1)
            end = start + max_length
            tokenized_batches.append(enc.input_ids[:, start:end].squeeze(0))

    print("number of samples:", len(tokenized_batches))
    return tokenized_batches


def _load_c4_from_arrow_cache():
    """Offline fallback when `load_dataset(allenai/c4)` fingerprints do not match HF_DATASETS_CACHE."""
    from datasets import Dataset, concatenate_datasets

    cache_root = os.environ.get(
        "HF_DATASETS_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets"),
    )
    base = os.path.join(cache_root, "allenai___c4")
    if not os.path.isdir(base):
        return None, None
    train_pattern = os.path.join(base, "default-*", "0.0.0", "*", "c4-train-*.arrow")
    val_pattern = os.path.join(base, "default-*", "0.0.0", "*", "c4-validation*.arrow")
    train_arrows = sorted(glob.glob(train_pattern))
    val_arrows = sorted(glob.glob(val_pattern))
    if not train_arrows or not val_arrows:
        return None, None
    train_data = concatenate_datasets([Dataset.from_file(p) for p in train_arrows])
    val_data = concatenate_datasets([Dataset.from_file(p) for p in val_arrows])
    return train_data, val_data


def c4(tokenizer, batch_size, train_samples, val_samples, gpt_samples, num_workers, max_length,
       shuffle_seed=1234, **_kw):
    try:
        train_data = load_dataset(
            'allenai/c4',
            data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
            split='train'
        )
        val_data = load_dataset(
            'allenai/c4',
            data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
            split='validation'
        )
    except ValueError as e:
        err = str(e)
        if "Couldn't find cache" not in err:
            raise
        train_data, val_data = _load_c4_from_arrow_cache()
        if train_data is None:
            raise RuntimeError(
                f"C4 dataset cache mismatch ({err}). Point HF_DATASETS_CACHE at a dir containing "
                f"allenai___c4/*/c4-train-*.arrow and c4-validation*.arrow, or refresh the dataset online."
            ) from e

    train_tokens = tokenize_texts(train_data, tokenizer, max_length, train_samples + gpt_samples, seed=shuffle_seed)
    val_tokens = tokenize_texts(val_data, tokenizer, max_length, val_samples)

    train_loader = DataLoader(train_tokens[:train_samples], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_tokens, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    gpt_loader = DataLoader(train_tokens[train_samples:], batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, gpt_loader


def fineweb_edu(tokenizer, batch_size, train_samples, val_samples, gpt_samples, num_workers, max_length,
                shuffle_seed=1234, shuffle_buffer_size=100_000, seed=42, **_kw):
    all_data = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        "sample-10BT",
        split='train',
        streaming=True
    )

    all_data = all_data.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    
    all_tokens = make_concat_chunks(all_data, tokenizer, max_length, train_samples + val_samples + gpt_samples, seed=shuffle_seed)

    train_split = _maybe_shard(all_tokens[:train_samples])
    val_split = _maybe_shard(all_tokens[train_samples:train_samples+val_samples])
    gpt_split = _maybe_shard(all_tokens[train_samples+val_samples:])

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    gpt_loader = DataLoader(gpt_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, gpt_loader

def _broadcast_chunks(chunks, rank, world_size):
    """Broadcast pre-tokenized chunks from rank 0 to all ranks."""
    import torch.distributed as dist
    if world_size <= 1:
        return chunks

    if rank == 0:
        stacked = torch.stack(chunks)
    else:
        stacked = None

    n = torch.tensor(len(chunks) if rank == 0 else 0, dtype=torch.long, device="cuda")
    dist.broadcast(n, src=0)
    seq_len = torch.tensor(chunks[0].shape[0] if rank == 0 and len(chunks) > 0 else 0,
                           dtype=torch.long, device="cuda")
    dist.broadcast(seq_len, src=0)

    if rank != 0:
        stacked = torch.empty((n.item(), seq_len.item()), dtype=torch.long, device="cuda")
    else:
        stacked = stacked.to("cuda")

    dist.broadcast(stacked, src=0)
    return [stacked[i].cpu() for i in range(stacked.shape[0])]


def open_thoughts(tokenizer, batch_size, train_samples, val_samples, gpt_samples, num_workers, max_length,
                  shuffle_seed=1234, seed=42, open_thoughts_max_samples=10_000, **_kw):
    rank, world_size = _get_dist_info()

    tmpl = tokenizer.chat_template
    if tmpl is not None:
        tmpl = tmpl.replace(
            "<think></think>{{render_content(message)}}",
            "{%- set rc = message.get('reasoning_content', '') -%}"
            "<think>{{rc}}</think>{{render_content(message)}}"
        )
        tokenizer.chat_template = tmpl

    total_needed = train_samples + val_samples + gpt_samples

    if rank == 0 or world_size <= 1:
        ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
        ds = ds.shuffle(seed=seed).select(range(open_thoughts_max_samples))

        def preprocess(example):
            messages = []
            messages.append({
                "role": "system",
                "content": (
                    "You are Kimi, an AI assistant created by Moonshot AI."
                ),
            })
            for msg in example["conversations"]:
                role = msg["from"]
                if role == "user":
                    messages.append({"role": "user", "content": msg["value"]})
                else:
                    thought, solution = split_thought_solution(msg["value"])
                    messages.append({"role": "assistant", "content": solution, "reasoning_content": thought})

            return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

        print(f"Preprocessing {len(ds)} OpenThoughts samples (chat template)...", flush=True)
        ds = ds.map(preprocess, num_proc=min(8, os.cpu_count() or 1))
        print(f"Tokenizing into {total_needed} chunks of length {max_length}...", flush=True)
        all_chunks = make_concat_chunks(ds, tokenizer, max_length, total_needed, seed=shuffle_seed)
        print(f"Broadcasting {len(all_chunks)} chunks to {world_size} ranks...", flush=True)
    else:
        all_chunks = [torch.zeros(max_length, dtype=torch.long)]

    if world_size > 1:
        all_chunks = _broadcast_chunks(all_chunks, rank, world_size)
    if rank == 0:
        print("Dataset ready.", flush=True)

    train_tokens = all_chunks[:train_samples]
    val_tokens = all_chunks[train_samples:train_samples+val_samples]
    gpt_tokens = all_chunks[train_samples+val_samples:]

    train_split = _maybe_shard(train_tokens)
    val_split = _maybe_shard(val_tokens)
    gpt_split = _maybe_shard(gpt_tokens)

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    gpt_loader = DataLoader(gpt_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, gpt_loader

def mixed(tokenizer, batch_size, train_samples, val_samples, gpt_samples, num_workers, max_length, shuffle_seed=1234, seed=42, open_thoughts_max_samples=50_000, **_kw):
    rank, world_size = _get_dist_info()

    total_needed = train_samples + val_samples + gpt_samples

    if rank == 0 or world_size <= 1:
        ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
        ds = ds.shuffle(seed=seed)

        llm_c_tokens = make_concat_chunks(ds, tokenizer, max_length, 10000, seed=shuffle_seed)

        ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
        ds = ds.shuffle(seed=seed).select(range(open_thoughts_max_samples))

        tmpl = tokenizer.chat_template
        if tmpl is not None:
            tmpl = tmpl.replace(
                "<think></think>{{render_content(message)}}",
                "{%- set rc = message.get('reasoning_content', '') -%}"
                "<think>{{rc}}</think>{{render_content(message)}}"
            )
            tokenizer.chat_template = tmpl

        def preprocess(example):
            messages = []
            for msg in example["conversations"]:
                role = msg["from"]
                if role == "user":
                    messages.append({"role": "user", "content": msg["value"]})
                else:
                    thought, solution = split_thought_solution(msg["value"])
                    messages.append({"role": "assistant", "content": solution, "reasoning_content": thought})

            return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            }

        ds = ds.map(preprocess, remove_columns=ds.column_names, num_proc=min(8, os.cpu_count() or 1))

        ot_tokens = make_concat_chunks(ds, tokenizer, max_length, (total_needed - len(llm_c_tokens)) // 2, seed=shuffle_seed)

        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split='train', streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=30000)
        
        fw_tokens = make_concat_chunks(ds, tokenizer, max_length, (total_needed - len(llm_c_tokens) + 1) // 2, seed=shuffle_seed)

        all_chunks = llm_c_tokens + ot_tokens + fw_tokens
        random.shuffle(all_chunks)
        print(f"Broadcasting {len(all_chunks)} mixed chunks to {world_size} ranks...", flush=True)
    else:
        all_chunks = [torch.zeros(max_length, dtype=torch.long) for _ in range(total_needed)]

    if world_size > 1:
        all_chunks = _broadcast_chunks(all_chunks, rank, world_size)

    if rank == 0:
        print("Mixed dataset ready.", flush=True)

    train_tokens = all_chunks[:train_samples]
    val_tokens = all_chunks[train_samples:train_samples + val_samples]
    gpt_tokens = all_chunks[train_samples + val_samples:total_needed]

    train_split = _maybe_shard(train_tokens)
    val_split = _maybe_shard(val_tokens)
    gpt_split = _maybe_shard(gpt_tokens)

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    gpt_loader = DataLoader(gpt_split, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, gpt_loader
    

def patch_qwen35_chat_template(tokenizer):
    """
    Qwen3.5's template supports message.reasoning_content, but may emit
    an empty <think></think> block for non-thinking samples. This avoids
    that for historical assistant messages.
    """
    tmpl = getattr(tokenizer, "chat_template", None)
    if not tmpl:
        return tokenizer

    old = (
        "{%- if loop.index0 > ns.last_query_index %}\n\n"
        "{{- '<|im_start|>' + message.role + '\\n<think>\\n' "
        "+ reasoning_content + '\\n</think>\\n\\n' + content }}\n\n"
        "{%- else %}\n\n"
        "{{- '<|im_start|>' + message.role + '\\n' + content }}\n\n"
        "{%- endif %}"
    )

    new = (
        "{%- if loop.index0 > ns.last_query_index and reasoning_content %}\n\n"
        "{{- '<|im_start|>' + message.role + '\\n<think>\\n' "
        "+ reasoning_content + '\\n</think>\\n\\n' + content }}\n\n"
        "{%- else %}\n\n"
        "{{- '<|im_start|>' + message.role + '\\n' + content }}\n\n"
        "{%- endif %}"
    )

    if old in tmpl:
        tokenizer.chat_template = tmpl.replace(old, new)
    else:
        old_compact = "{%- if loop.index0 > ns.last_query_index %}"
        new_compact = "{%- if loop.index0 > ns.last_query_index and reasoning_content %}"
        if old_compact in tmpl and "reasoning_content" in tmpl and "<think>" in tmpl:
            tokenizer.chat_template = tmpl.replace(old_compact, new_compact, 1)

    return tokenizer


def patch_gemma4_chat_template(tokenizer):
    """
    Gemma4's template can know reasoning/reasoning_content, but the official
    condition may be too restrictive. This relaxes the condition so reasoning
    can be emitted for historical assistant messages.
    """
    tmpl = getattr(tokenizer, "chat_template", None)
    if not tmpl:
        return tokenizer

    old = (
        "{%- if thinking_text and loop.index0 > ns_turn.last_user_idx "
        "and message.get('tool_calls') -%}"
    )
    new = "{%- if thinking_text and loop.index0 > ns_turn.last_user_idx -%}"

    if old in tmpl:
        tokenizer.chat_template = tmpl.replace(old, new)

    return tokenizer

def normalize_reasoning_messages(
    example,
    *,
    include_system=True,
):
    """
    Converts one JSONL row into chat-template-compatible messages.

    Non-thinking:
      system + user + assistant final answer

    Thinking:
      system + user + assistant with reasoning_content + final answer
    """
    mode = example.get("mode")

    # Prefer calib_messages because they already contain assistant output.
    source_msgs = example.get("messages") or []

    messages = []

    for msg in source_msgs:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        content = msg.get("content", "")

        if content is None:
            content = ""

        if role == "system":
            if include_system:
                messages.append({"role": "system", "content": content})

        elif role == "user":
            messages.append({"role": "user", "content": content})

        elif role == "assistant":
            out = {"role": "assistant", "content": content}

            reasoning = example.get("reasoning")
            if mode == "thinking" and reasoning:
                out["reasoning_content"] = reasoning

            messages.append(out)

    # Fallback if calib_messages/messages did not include assistant.
    if not any(m["role"] == "assistant" for m in messages):
        output = example.get("final_output") or example.get("output")
        if not output:
            return None

        out = {"role": "assistant", "content": output}

        reasoning = example.get("reasoning")
        if mode == "thinking" and reasoning:
            out["reasoning_content"] = reasoning

        messages.append(out)

    return messages


def custom_jsonl_to_text_reasoning(
    example,
    tokenizer,
    *,
    include_system=True,
):
    messages = normalize_reasoning_messages(
        example,
        include_system=include_system,
    )

    if messages is None:
        return {"text": None}

    enable_thinking = example.get("mode") == "thinking"

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )

    return {"text": text}


def custom_jsonl(
    tokenizer,
    batch_size,
    train_samples,
    val_samples,
    gpt_samples,
    num_workers,
    max_length,
    *,
    model_name="qwen3.5-4b",
    patch_tokenizer_fn=patch_qwen35_chat_template,
    shuffle_seed=1234,
    seed=42,
    nonthinking_weight=1.0,
    thinking_weight=1.0,
    include_system=True,
    **_kw,
):
    rank, world_size = _get_dist_info()
    total_needed = train_samples + val_samples + gpt_samples

    if patch_tokenizer_fn is not None:
        tokenizer = patch_tokenizer_fn(tokenizer)

    if rank == 0 or world_size <= 1:
        ds_nt = load_dataset(
            "json",
            data_files="calib_nonthinking_clean.jsonl",
            split="train",
        )

        ds_th = load_dataset(
            "json",
            data_files="calib_thinking_clean.jsonl",
            split="train",
        )

        ds_nt = ds_nt.shuffle(seed=seed)
        ds_th = ds_th.shuffle(seed=seed + 1)

        def preprocess(ex):
            return custom_jsonl_to_text_reasoning(
                ex,
                tokenizer,
                include_system=include_system,
            )

        num_proc = min(8, os.cpu_count() or 1)

        ds_nt = ds_nt.map(
            preprocess,
            remove_columns=ds_nt.column_names,
            num_proc=num_proc,
        ).filter(lambda x: x["text"] is not None)

        print(ds_nt[0])

        ds_th = ds_th.map(
            preprocess,
            remove_columns=ds_th.column_names,
            num_proc=num_proc,
        ).filter(lambda x: x["text"] is not None)

        print(ds_th[0])

        w_sum = nonthinking_weight + thinking_weight
        nt_needed = int(total_needed * nonthinking_weight / w_sum)
        th_needed = total_needed - nt_needed

        nt_chunks = make_concat_chunks(
            ds_nt,
            tokenizer,
            max_length,
            nt_needed,
            seed=shuffle_seed,
            add_eos=True,
        )

        th_chunks = make_concat_chunks(
            ds_th,
            tokenizer,
            max_length,
            th_needed,
            seed=shuffle_seed,
            add_eos=True,
        )

        all_chunks = nt_chunks + th_chunks
        random.Random(shuffle_seed).shuffle(all_chunks)

        if len(all_chunks) < total_needed:
            raise ValueError(
                f"Only produced {len(all_chunks)} chunks, but needed {total_needed}. "
                f"Non-thinking chunks: {len(nt_chunks)}, thinking chunks: {len(th_chunks)}."
            )

        all_chunks = all_chunks[:total_needed]

        print(
            f"Broadcasting {len(all_chunks)} custom {model_name} chunks "
            f"to {world_size} ranks...",
            flush=True,
        )

    else:
        all_chunks = [torch.zeros(max_length, dtype=torch.long)]

    if world_size > 1:
        all_chunks = _broadcast_chunks(all_chunks, rank, world_size)

    if rank == 0:
        print(f"Custom {model_name} dataset ready.", flush=True)

    train_tokens = all_chunks[:train_samples]
    val_tokens = all_chunks[train_samples:train_samples + val_samples]
    gpt_tokens = all_chunks[train_samples + val_samples:total_needed]

    train_split = _maybe_shard(train_tokens)
    val_split = _maybe_shard(val_tokens)
    gpt_split = _maybe_shard(gpt_tokens)

    train_loader = DataLoader(
        train_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    gpt_loader = DataLoader(
        gpt_split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, gpt_loader

def create_dataloader(dataset_name, tokenizer, batch_size, train_samples, val_samples, gpt_samples, num_workers, max_length, **kwargs):
    return globals()[dataset_name](tokenizer, batch_size, train_samples, val_samples, gpt_samples, num_workers, max_length, **kwargs)
