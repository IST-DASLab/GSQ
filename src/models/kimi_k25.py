import time
import torch
import math
import os, gc
from accelerate.utils import set_module_tensor_to_device
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from .base import BaseModelWrapper
from src.evaluation.wiki_eval import *
from src.utils.progress_reporter import report_ppl_layer

class KimiK25Wrapper(BaseModelWrapper):
    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype, dummy=False):
        super().__init__(model_name, tokenizer, batch_size, seqlen, device, dtype, dummy=dummy)
        self.quantization_config = self.dict_to_ns(self.model.language_model.config.quantization_config)
        self.layer_prefix = "language_model.model.layers"
        self.num_layers = len(self.model.language_model.model.layers)
        self.num_experts = self.model.language_model.config.n_routed_experts
        self.is_moe = True

    def _set_tensors(self, name_shard_pairs):
        if self.dummy:
            self._materialize_to_device(name_shard_pairs)
            return

        by_shard = {}
        for n, s in name_shard_pairs:
            by_shard.setdefault(s, []).append(n)

        for shard, names in by_shard.items():
            tensors = safe_load_file(shard, device=self.device)
            for n in names:
                if n.endswith(".weight_shape") or n.endswith(".weight_scale") or n.endswith("inv_freq"):
                    continue
                if n.endswith(".weight_packed"):
                    base = n[: -len(".weight_packed")]
                    compressed_data = {
                        "weight_packed": tensors[f"{base}.weight_packed"], 
                        "weight_scale": tensors[f"{base}.weight_scale"],
                        "weight_shape": tensors[f"{base}.weight_shape"]
                    }
                    W_deq = self.compressor.decompress_weight(compressed_data, self.quantization_config.config_groups.group_0.weights)

                    set_module_tensor_to_device(self.model, f"{base}.weight", self.device, value=W_deq)
                    continue

                set_module_tensor_to_device(self.model, n, self.device, value=tensors[n])

    def _build_layer_inputs(self, batch_size):
        # DeepseekV3FlashAttention2 expects attention_mask=None (it uses causal=True
        # internally). Passing a 4D causal mask triggers the unpadding path which
        # expects a 2D padding mask and would silently fall back to eager attention.
        inputs = dict(self.kwargs)
        inputs["attention_mask"] = None
        return inputs

    def _layer_prefixes(self, layer_name):
        layer_idx = int(layer_name.split('.')[-1])
        base = f"{self.layer_prefix}.{layer_idx}"
        if layer_idx < self.model.language_model.config.first_k_dense_replace:
            non_mlp = [
                f"{base}.input_layernorm",
                f"{base}.self_attn",
                f"{base}.post_attention_layernorm"
            ]
            mlp = [
                f"{base}.mlp"
            ]
        else:
            non_mlp = [
                f"{base}.input_layernorm",
                f"{base}.self_attn",
                f"{base}.mlp.gate",
                f"{base}.mlp.shared_experts",
                f"{base}.post_attention_layernorm"
            ]
            mlp = [
                f"{base}.mlp.experts.{e}"
                for e in range(self.num_experts)
            ]
        return {"non_mlp": non_mlp, "mlp": mlp}

    def move_embed_to(self, device):
        names = self._names_from_ckpt(["language_model.model.embed_tokens"])
        if device == "cuda":
            self._set_tensors(names)
        else:
            self._offload_names_to_meta(names)

    def move_output_heads_to(self, device):
        names = []
        names += self._names_from_ckpt("language_model.model.norm")
        names += self._names_from_ckpt("language_model.lm_head")
        if device == "cuda":
            self._set_tensors(names)
        else:
            self._offload_names_to_meta(names)

    def _offload_names_to_meta(self, name_shard_pairs):
        names = [n if isinstance(n, str) else n[0] for n in name_shard_pairs]
        for n in names:
            if n.endswith(".weight_shape") or n.endswith(".weight_scale") or n.endswith("inv_freq"):
                continue
            if n.endswith(".weight_packed"):
                base = n[: -len(".weight_packed")]
                set_module_tensor_to_device(self.model, f"{base}.weight", "meta")
                continue
            
            set_module_tensor_to_device(self.model, n, "meta")
        torch.cuda.empty_cache()

    def offload_non_experts_to_meta(self, layer_name, exclude=["post_attention_layernorm", "mlp.gate", "mlp.shared_experts"]):
        prefixes = self._layer_prefixes(layer_name)["non_expert"]
        prefixes = [p for p in prefixes if not any(x in p for x in exclude)]
        pairs = self._names_from_ckpt(prefixes)
        self._offload_names_to_meta(pairs)

    def get_mlp_input(self, batch):
        current_layer = self.get_layer_module(self.current_layer_idx)

        additional_layer_inputs = self._build_layer_inputs(batch.shape[0])
        hidden_states = current_layer.input_layernorm(batch)
        hidden_states, _, _ = current_layer.self_attn(hidden_states, **additional_layer_inputs)
        return hidden_states + batch
    
    def get_mlp_output(self, mlp_input_batch):
        current_layer = self.get_layer_module(self.current_layer_idx)

        hidden_states = current_layer.post_attention_layernorm(mlp_input_batch)
        hidden_states = current_layer.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        return hidden_states + mlp_input_batch

    def get_layer_module(self, idx):
        return self.model.language_model.model.layers[idx]
    
    def update_quantized_weights(self, layer_name, quantized_weights):
        layer = self._get_layer_by_name(layer_name)
        if isinstance(quantized_weights, tuple):
            Q, scales = quantized_weights
            self.temp_weights[layer_name] = layer.weight.data
            self.temp_weights[f"{layer_name}.scale"] = scales
            with torch.no_grad():
                layer.weight.data = Q.to(layer.weight.device).to(layer.weight.dtype)
        else:
            with torch.no_grad():
                if ".experts" in layer_name:
                    layer.weight.data = quantized_weights.to(layer.weight.device)
                else:
                    layer.weight.data = quantized_weights.to(layer.weight.dtype).to(layer.weight.device)
    
    @torch.no_grad()
    def ppl_evaluation(self, read_from_disk=-1):
        dataset = get_dataset("wikitext2")
        testloader = prepare_test_dataloader(
                dataset=dataset["test"], 
                tokenizer=self.tokenizer, 
                seqlen=self.model.seqlen,
                batch_size=4
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

        additional_layer_inputs = {"attention_mask": None}
        for k, v in self.kwargs.items():
            additional_layer_inputs[k] = v

        idx_copy = self.current_layer_idx
        ppl_start = time.time()

        for i in range(self.num_layers):
            if self.rank == 0:
                report_ppl_layer(i, self.num_layers, elapsed=time.time() - ppl_start)
            self.current_layer_idx = i
            layer_name = f"{self.layer_prefix}.{i}"
            if i <= read_from_disk and i >= self.model.language_model.config.first_k_dense_replace:
                self.load_from_disc(layer_name)
            else:
                self.move_layer_to_gpu(layer_name)
            layer = self.get_layer_module(i)

            for b in range(num_batches):
                ids_cpu = input_ids_cpu_list[b]

                x_cpu = activations_cpu_list[b]
                if x_cpu is None:
                    ids = ids_cpu.to(self.device, non_blocking=True)
                    x = self.model.language_model.model.embed_tokens(ids)
                else:
                    x = x_cpu.to(self.device, non_blocking=True)

                hidden = layer.input_layernorm(x)
                attn_out, _, _ = layer.self_attn(hidden, **additional_layer_inputs)
                x = x + attn_out

                x = self.get_mlp_output(x)

                activations_cpu_list[b] = x.to("cpu", non_blocking=True).pin_memory()

                del x, hidden, attn_out

            self.offload_to_meta(layer_name)
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

        local_nll_sum = torch.tensor(0.0, device=self.device)
        local_tok_cnt = torch.tensor(0.0, device=self.device)

        for b in range(num_batches):
            ids_cpu = input_ids_cpu_list[b]
            x_cpu = activations_cpu_list[b]

            input_ids = ids_cpu.to(self.device, non_blocking=True)
            x = x_cpu.to(self.device, non_blocking=True)

            x = self.model.language_model.model.norm(x)
            logits = self.model.language_model.lm_head(x)

            logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]

            nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()
            mask = shift_labels != loss_fn.ignore_index
            nll = (nll * mask).sum(dim=1)
            tok = mask.sum(dim=1)

            local_nll_sum += nll.sum()
            local_tok_cnt += tok.sum()

            del input_ids, x, x_cpu, logits, shift_labels, nll, tok

        self.move_embed_to("meta")
        self.move_output_heads_to("meta")

        self.current_layer_idx = idx_copy

        mean_nll = (local_nll_sum / local_tok_cnt).item()
        ppl = math.exp(mean_nll)

        torch.cuda.synchronize(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        return ppl
        
    def save_prefixes_to_disc(self, prefixes, config=None):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        if not prefixes:
            return

        os.makedirs(self.save_dir, exist_ok=True)

        for pfx in prefixes:
            module = self.model.get_submodule(pfx)

            sd = module.state_dict(keep_vars=True)
            to_save = {}

            for local_name, tensor in sd.items():
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    if ".expert" in pfx:
                        weight = tensor.detach().cpu()
                        self.quantization_config.config_groups.group_0.weights.group_size = self.groupsize
    
                        flat = weight.flatten()
                        N = flat.numel()
                        num_blocks = N // self.groupsize
                        flat = flat[:num_blocks * self.groupsize]
                        blocks = flat.view(-1, self.groupsize)
                        abs_blocks = blocks.abs()
                        nonzero_mask = abs_blocks > 0

                        abs_blocks_masked = abs_blocks.clone()
                        abs_blocks_masked[~nonzero_mask] = float('inf')

                        scale, _ = abs_blocks_masked.min(dim=1)
                        scale[scale == float('inf')] = 0.0
                        scale = scale.view(weight.shape[0], -1)
                       
                        compressed = self.compressor.compress_weight(
                            weight=weight,
                            scale=scale,
                            quantization_args=self.quantization_config.config_groups.group_0.weights
                        )
                        new_name = local_name[:-len(".weight")]
                        base = f"{pfx}.{new_name}"
                        to_save[base + ".weight_packed"] = compressed["weight_packed"]
                        to_save[base + ".weight_scale"]  = scale
                        to_save[base + ".weight_shape"]  = compressed["weight_shape"]
                    else:
                        to_save[f"{pfx}.{local_name}"] = tensor.detach().cpu()

            safe = pfx.replace('.', '_')
            path = os.path.join(self.save_dir, f"{safe}.safetensors")
            safe_save_file(to_save, path)
