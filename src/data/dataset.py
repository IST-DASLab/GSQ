import os
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

def c4(tokenizer, batch_size, train_samples, val_samples, gpt_samples, num_workers, max_length,
       shuffle_seed=1234, **_kw):
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

def create_dataloader(dataset_name, tokenizer, batch_size, train_samples, val_samples, gpt_samples, num_workers, max_length, **kwargs):
    return globals()[dataset_name](tokenizer, batch_size, train_samples, val_samples, gpt_samples, num_workers, max_length, **kwargs)
