import os
import time
import torch
import torch.nn.functional as F
import math
import gc
import torch.distributed as dist
from safetensors.torch import save_file as safe_save_file
from transformers.models.gemma4.modeling_gemma4 import create_causal_mask, create_sliding_window_causal_mask
from .base import BaseModelWrapper
from src.evaluation.wiki_eval import *
from src.utils.progress_reporter import report_ppl_layer

class Gemma4Wrapper(BaseModelWrapper):
    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype):
        super().__init__(model_name, tokenizer, batch_size, seqlen, device, dtype)
        self.layer_prefix = "model.language_model.layers"
        self.num_layers = len(self.model.model.layers)

    @torch.no_grad()
    def get_inputs(self, data_dict, data_loader):
        current_layer = self.get_layer_module(self.current_layer_idx)
        cache = {'index': 0}
        def store_input_hook(_, args, kwargs):
            start = cache['index'] * self.batch_size
            end = min(start + self.batch_size, data_dict['input'].shape[0])
            if isinstance(args, tuple):
                args = args[0]
            data_dict['input'][start:end] = args
            cache['index'] += 1
            for k, v in kwargs.items():
                if k == "attention_mask":
                    if v is not None:
                        self._attention_mask_1 = v[:1]
                elif k not in ("hidden_states", "past_key_values", "past_key_value"):
                    self.kwargs[k] = v
            raise ValueError
            
        total_batches = len(data_loader)
        handle = current_layer.register_forward_pre_hook(store_input_hook, with_kwargs=True)
        for batch_idx, batch in enumerate(data_loader):
            try:
                self.model(batch.to(self.device))
            except ValueError:
                pass
            if self.rank == 0 and (batch_idx + 1) % max(1, total_batches // self.batch_report_divisor) == 0:
                print(f"  get_inputs: {batch_idx + 1}/{total_batches} batches", flush=True)
        handle.remove()

        self.position_embeddings = {}
        position_ids = torch.arange(data_dict['input'].shape[1], device=self.device).unsqueeze(0)
        for layer_type in self.model.model.unique_layer_types:
            self.position_embeddings[layer_type] = self.model.model.rotary_emb(data_dict['input'][0:1].to(self.device), position_ids, layer_type)

        mask_kwargs = {
            "config": self.model.config,
            "inputs_embeds": data_dict['input'][0:1].to(self.device),
            "attention_mask": None,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        self.causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs).to(self.device),
        }

    def _build_layer_inputs(self, batch_size=None):
        inputs = dict(self.kwargs)
        mask = self.causal_mask_mapping[
            self.model.config.layer_types[self.current_layer_idx]
        ]
        if batch_size is not None and mask is not None:
            mask = mask.expand(batch_size, *mask.shape[1:])
        if mask is not None:
            inputs["attention_mask"] = self._attention_mask_1
        else:
            inputs["attention_mask"] = None 
        inputs["position_embeddings"] = self.position_embeddings[
            self.model.config.layer_types[self.current_layer_idx]
        ]
        inputs["shared_kv_states"] = {}
        return inputs
    
    def get_mlp_input(self, batch):
        current_layer = self.get_layer_module(self.current_layer_idx)

        additional_layer_inputs = self._build_layer_inputs()

        hidden_states = current_layer.input_layernorm(batch)
        hidden_states, _ = current_layer.self_attn(hidden_states, **additional_layer_inputs)
        hidden_states = current_layer.post_attention_layernorm(hidden_states)
        return hidden_states + batch
    
    def get_mlp_output(self, mlp_input_batch):
        current_layer = self.get_layer_module(self.current_layer_idx)

        hidden_states = current_layer.pre_feedforward_layernorm(mlp_input_batch)
        hidden_states = current_layer.mlp(hidden_states)
        hidden_states = current_layer.post_feedforward_layernorm(hidden_states)
        return (hidden_states + mlp_input_batch) * current_layer.layer_scalar

    def get_layer_module(self, idx):
        return self.model.model.layers[idx]
    
    def _layer_prefixes(self, layer_name):
        layer_idx = int(layer_name.split('.')[-1])
        base = f"{self.layer_prefix}.{layer_idx}"
        non_mlp = [
            f"{base}.input_layernorm",
            f"{base}.self_attn",
            f"{base}.post_attention_layernorm",
            f"{base}.pre_feedforward_layernorm",
            f"{base}.post_feedforward_layernorm",
            f"{base}.layer_scalar"
        ]
        mlp = [
            f"{base}.mlp"
        ]
        return {"non_mlp": non_mlp, "mlp": mlp}

    def move_embed_to(self, device):
        names = self._names_from_ckpt(["model.language_model.embed_tokens"])
        if device == "cuda":
            self._set_tensors(names)
        else:
            self._offload_names_to_meta(names)

    def move_output_heads_to(self, device):
        names = []
        names += self._names_from_ckpt("model.language_model.norm")
        names += self._names_from_ckpt("lm_head")
        if device == "cuda":
            self._set_tensors(names)
        else:
            self._offload_names_to_meta(names)

    def save_attention_to_disc(self, pfx):
        os.makedirs(self.save_dir, exist_ok=True)
        to_save = {}
        if self.model.config.layer_types[self.current_layer_idx] == "full_attention":
            compressed_attn_layers = ['q_proj', 'k_proj', 'o_proj']
        else:
            compressed_attn_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        non_compressed_attn_layers = ['k_norm', 'q_norm']
        self.quantization_config.config_groups.group_0.weights.group_size = self.groupsize
        for name in compressed_attn_layers:
            base = f"{pfx}.{name}"
            module = self.model.get_submodule(base.replace(".language_model", ""))
            if isinstance(self.temp_weights[f"{base}.scale"], tuple):
                self.temp_weights[f"{base}.scale"] = self.temp_weights[f"{base}.scale"][0]
            compressed = self.compressor.compress_weight(
                weight=module.weight.data.detach().cpu(),
                scale=self.temp_weights[f"{base}.scale"].detach().cpu(),
                quantization_args=self.quantization_config.config_groups.group_0.weights
            )
            to_save[base + ".weight_packed"] = compressed["weight_packed"]
            to_save[base + ".weight_scale"]  = self.temp_weights[f"{base}.scale"].detach().cpu()
            to_save[base + ".weight_shape"]  = compressed["weight_shape"]
        for name in non_compressed_attn_layers:
            base = f"{pfx}.{name}"
            module = self.model.get_submodule(base.replace(".language_model", ""))
            to_save[base + ".weight"] = module.weight.data.detach().cpu()
        safe = pfx.replace('.', '_')
        path = os.path.join(self.save_dir, f"{safe}.safetensors")
        safe_save_file(to_save, path)

    def save_prefixes_to_disc(self, prefixes, exclude=[]):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        if not prefixes:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        if "self_attn" in exclude:
            current_layer = self.get_layer_module(self.current_layer_idx)
            if getattr(current_layer, "layer_type", None) == "linear_attn":
                exclude.append("linear_attn")
        prefixes = [p for p in prefixes if not any(x in p for x in exclude)]
        for pfx in prefixes:
            if "layer_scalar" in pfx:
                to_save = {}
                to_save[pfx] = current_layer.layer_scalar
                safe = pfx.replace('.', '_')
                path = os.path.join(self.save_dir, f"{safe}.safetensors")
                safe_save_file(to_save, path)
                continue

            module = self.model.get_submodule(pfx.replace(".language_model", ""))
            sd = module.state_dict(keep_vars=True)
            to_save = {}
            for local_name, tensor in sd.items():
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    to_save[f"{pfx}.{local_name}"] = tensor.detach().cpu()
            safe = pfx.replace('.', '_')
            path = os.path.join(self.save_dir, f"{safe}.safetensors")
            safe_save_file(to_save, path)

    @torch.no_grad()
    def ppl_evaluation(self, read_from_disk=-1):
        dataset = get_dataset("wikitext2", self.tokenizer)
        testloader = prepare_test_dataloader(
                dataset=dataset["test"], 
                tokenizer=self.tokenizer, 
                seqlen=self.model.seqlen,
                batch_size=8,
                world_size=self.world_size,
                rank=self.rank
        )
        
        pad_token_id = self.model.config.pad_token_id

        if pad_token_id is not None:
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        self.model.eval()

        self.move_embed_to(self.device)
        self.move_output_heads_to(self.device)

        input_ids_cpu_list = []
        activations_cpu_list = []

        for batch in testloader:
            ids_cpu = batch["input_ids"].to("cpu", non_blocking=True).pin_memory()
            input_ids_cpu_list.append(ids_cpu)
            activations_cpu_list.append(None)
            del batch

        num_batches = len(input_ids_cpu_list)

        idx_copy = self.current_layer_idx
        ppl_start = time.time()

        for i in range(self.num_layers):
            if self.rank == 0:
                report_ppl_layer(i, self.num_layers, elapsed=time.time() - ppl_start)
            self.current_layer_idx = i
            layer_name = f"{self.layer_prefix}.{i}"
            if i <= read_from_disk:
                self.load_from_disc(layer_name)
            else:
                self.move_layer_to_gpu(layer_name)
            layer = self.get_layer_module(i)

            for b in range(num_batches):
                ids_cpu = input_ids_cpu_list[b]

                x_cpu = activations_cpu_list[b]
                if x_cpu is None:
                    ids = ids_cpu.to(self.device, non_blocking=True)
                    x = self.model.model.embed_tokens(ids)
                else:
                    x = x_cpu.to(self.device, non_blocking=True)

                additional_layer_inputs = self._build_layer_inputs(x.shape[0])
                x = layer(x, **additional_layer_inputs)
                #x = self.get_mlp_input(x)
                #x = self.get_mlp_output(x)

                activations_cpu_list[b] = x.to("cpu", non_blocking=True).pin_memory()

            self.offload_to_meta(layer_name)
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

        nll_sum = torch.tensor(0.0, device=self.device)
        tok_cnt = torch.tensor(0.0, device=self.device)

        self.model.tie_weights()

        for b in range(num_batches):
            ids_cpu = input_ids_cpu_list[b]
            x_cpu = activations_cpu_list[b]

            input_ids = ids_cpu.to(self.device, non_blocking=True)
            x = x_cpu.to(self.device, non_blocking=True)

            x = self.model.model.norm(x)
            logits = self.model.lm_head(x)
            
            cap = self.model.config.final_logit_softcapping
            if cap is not None:
                logits = torch.tanh(logits / cap) * cap

            logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()
            mask = shift_labels != loss_fn.ignore_index
            nll = (nll * mask).sum(dim=1)
            tok = mask.sum(dim=1)

            nll_sum += nll.sum()
            tok_cnt += tok.sum()

        self.move_embed_to("meta")
        self.move_output_heads_to("meta")

        self.current_layer_idx = idx_copy

        if self.world_size > 1:
            dist.all_reduce(nll_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(tok_cnt, op=dist.ReduceOp.SUM)

        mean_nll = (nll_sum / tok_cnt).item()
        ppl = math.exp(mean_nll)

        torch.cuda.synchronize(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        return ppl
