import numpy as np
import random, argparse
import torch, transformers, datasets
import tqdm, pprint, logging
from torch.utils.data import DataLoader, Dataset, Subset
import re


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

def map_tensors(obj,  device, dtype=None):
    """Recursively map tensors to device and dtype."""
    if isinstance(obj, torch.Tensor):
        if device is not None:
            obj = obj.to(device=device)
        if dtype is not None:
            obj = obj.to(dtype=dtype)
        return obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(map_tensors(x, device, dtype) for x in obj)
    elif isinstance(obj, dict):
        return {k: map_tensors(v, device, dtype) for k, v in obj.items()}  # type: ignore
    else:
        return obj

def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization! 
    pass


def prepare_test_dataloader(
    dataset: datasets.Dataset, 
    tokenizer: transformers.PreTrainedTokenizerBase, 
    seqlen: int = 2048, 
    batch_size: int = 1,
    world_size: int = 1,
    rank: int = 0
) -> DataLoader[dict[str, torch.Tensor]]:
    """
    Get a DataLoader from a test dataset. This dataloader should be used when comparing WikiText2 perplexities with other papers, e.g. SparseGPT (arxiv.org/abs/2301.00774).

    Args:
        dataset: The dataset to create a dataloader from.
        tokenizer: The tokenizer to use.
        seqlen: The sequence length of sequences in the dataset.
        batch_size: The batch size.

    Returns:
        A DataLoader.
    """

    print(f"Preparing test dataloader")

    class TestDataset(Dataset):
        def __init__(self, ds, tokenizer, seqlen=2048):
            """Tokenize the entire dataset and reshape it into sequences of length seqlen."""
            tokenized_ds = tokenizer("\n\n".join(ds['text']), return_tensors='pt')
            nsamples = tokenized_ds.input_ids.numel() // seqlen
            # nsamples = 128

            input_ids = tokenized_ds.input_ids[0, : nsamples * seqlen]
            input_ids = input_ids.reshape(nsamples, seqlen)
            attn_mask = tokenized_ds.attention_mask[0, : nsamples * seqlen]
            attn_mask = attn_mask.reshape(nsamples, seqlen)

            self.input_ids = input_ids
            self.attn_mask = attn_mask

        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx], "attention_mask": self.attn_mask[idx]}

        def __len__(self):
            return len(self.input_ids)

    test_ds = TestDataset(dataset, tokenizer, seqlen)
    n = len(test_ds)
    per_rank = n // world_size
    shard = Subset(test_ds, list(range(rank * per_rank, (rank + 1) * per_rank)))
    loader = DataLoader(shard, batch_size=batch_size)
    print(f"Preparing test dataloader done")
    return loader


def get_dataset(name: str, tokenizer, ppl_seed=1234, ppl_max_samples=1000, test_size=0.2, split_seed=42) -> datasets.DatasetDict:
    """
    Get the dataset from the HuggingFace datasets library.

    Args:
        name: The name of the HuggingFace dataset to load. Must be one of "wikitext2", "ptb", "c4" or "alpaca".

    Returns:
        The dataset.
    """
    print(f"Loading dataset: {name}")

    if name == "open_thoughts":
        return open_thoughts_test_dataset(tokenizer, ppl_seed=ppl_seed, ppl_max_samples=ppl_max_samples)

    ds_properties = {
        "wikitext2": {"path": "wikitext", "config_name": "wikitext-2-raw-v1"},
        "ptb": {"path": "ptb_text_only", "config_name": "penn_treebank"},
        "c4": {
            "path": "allenai/c4",
            "config_name": "allenai--c4",
            "data_files": {
                "train": "en/c4-train.00000-of-01024.json.gz",
                "validation": "en/c4-validation.00000-of-00008.json.gz",
            },
            "cols_to_remove": ['url', 'timestamp'],
        },
        "alpaca": {"path": "tatsu-lab/alpaca", "cols_to_remove": ['input', 'output', 'instruction']},
    }

    if name not in ds_properties:
        raise NotImplementedError("The provided dataset is not supported")

    properties = ds_properties[name]
    ds = datasets.load_dataset(
        properties["path"], name=properties.get("config_name"), data_files=properties.get("data_files")
    )

    if "cols_to_remove" in properties:
        ds = ds.remove_columns(properties["cols_to_remove"])

    if name == "alpaca":
        ds = ds["train"].train_test_split(test_size=test_size, seed=split_seed)
        temp_ds = ds.pop("test")
        temp_ds = temp_ds.train_test_split(test_size=0.5, seed=split_seed)
        ds["test"] = temp_ds["train"]
        ds["validation"] = temp_ds["test"]

    print("Loading dataset done")
    return ds


@torch.no_grad()
def evaluate_ppl(
    model, 
    pad_token_id,
    testloader
) -> float:
    """
    Evaluate the model's perplexity on the test set using batch processing.
    It is expected that model is already on the correct device.
    """


    model.eval()

    if pad_token_id:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    nlls = []

    logging.info("Evaluating perplexity...")
    for batch in tqdm.tqdm(testloader, desc="Evaluating perplexity", unit="batch"):
        logging.debug(f"Evaluating batch {len(nlls)}")
        batch = map_tensors(batch, 'cuda:0')
        
        logits = model(**batch).logits

        # shift outputs and labels autoregressively.
        logits = logits[:, :-1, :]
        shift_labels = batch["input_ids"][:, 1:]

        # CrossEntropyLoss demands data dimension is dimension 1.
        nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()

        mask = shift_labels != loss_fn.ignore_index
        nll_means = (nll * mask).sum(dim=1) / mask.sum(dim=1)
        nlls.append(nll_means)

    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())


    return ppl.item()

def split_thought_solution(text: str):
    thought_re = re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.DOTALL)
    solution_re = re.compile(r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", re.DOTALL)

    thought = thought_re.search(text).group(1).strip()
    solution = solution_re.search(text).group(1).strip()

    return thought, solution

def open_thoughts_test_dataset(tokenizer, ppl_seed=1234, ppl_max_samples=1000):
    ds = datasets.load_dataset("open-thoughts/OpenThoughts-114k", split="train")
    ds = ds.shuffle(seed=ppl_seed).select(range(ppl_max_samples))

    def preprocess(example):
        messages = []
        for msg in example["conversations"]:
            role = msg["from"]
            if role == "user":
                messages.append({"role": "user", "content": msg["value"]})
            else:
                thought, solution = split_thought_solution(msg["value"])
                messages.append({"role": "assistant", "content": solution, "reasoning_content": thought})

        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    ds = ds.map(preprocess)
    return {"test": ds}
