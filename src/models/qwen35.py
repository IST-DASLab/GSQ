import os
import time
import math
import gc
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import ExitStack
from safetensors.torch import save_file as safe_save_file

from .base import BaseModelWrapper
from src.distributed import row_parallel_sum
from src.evaluation.wiki_eval import *
from src.prior.gptq import *
from src.utils.progress_reporter import report_gptq_calib, report_gptq_linear, report_ppl_layer


class Qwen35Wrapper(BaseModelWrapper):
    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype):
        super().__init__(model_name, tokenizer, batch_size, seqlen, device, dtype)

        self.layer_prefix = "model.language_model.layers"
        self.num_layers = len(self.model.model.layers)

    def get_layer_module(self, idx):
        return self.model.model.layers[idx]
    
    def get_layer_initialization(self, trainer, gpt_all, config, logging, is_attn=False):
        if logging is not None:
            logging = logging.logger
        current_layer = self.get_layer_module(self.current_layer_idx)
        self.groupsize = config.quantization.groupsize
        subset = {}
        
        if is_attn:
            if current_layer.layer_type != "full_attention":
                base_prefix = f"{self.get_current_layer()}.linear_attn"
                subset[f"{base_prefix}.in_proj_qkv"] = current_layer.linear_attn.in_proj_qkv
                subset[f"{base_prefix}.in_proj_z"] = current_layer.linear_attn.in_proj_z
                subset[f"{base_prefix}.out_proj"] = current_layer.linear_attn.out_proj
            else:
                base_prefix = f"{self.get_current_layer()}.self_attn"
                for name, module in current_layer.self_attn.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        subset[f"{base_prefix}.{name}"] = module
        else:
            base_prefix = f"{self.get_current_layer()}.mlp"
            for name, module in current_layer.mlp.named_modules():
                if isinstance(module, torch.nn.Linear):
                    subset[f"{base_prefix}.{name}"] = module
        
        skip_gptq = getattr(config.gptq, 'skip', False)

        if skip_gptq:
            for name in subset:
                Q, scales = rtn_quantize(subset[name], config, self.device, self.dtype)
                if "q_proj" in name or "k_proj" in name or "in_proj_qkv" in name:
                    self.update_quantized_weights(name, (Q, scales))
                    if self.world_size > 1:
                        layer = self._get_layer_by_name(name)
                        dist.broadcast(layer.weight.data, src=0)
                else:
                    trainer.setup_layer_training(name, Q, scales)
            return

        gpts = {}
        for name in subset:
            gpts[name] = GPTQ(subset[name], name, config, self.device, self.dtype)
            if config.gptq.wbits < 16:
                gpts[name].quantizer = Quantizer()
                if config.quantization.gsq_bits == "nvfp4":
                    gpts[name].quantizer.configure(
                        4, perchannel=True, sym=True, mse=True, trits=False, format="nvfp4"
                    )
                else:
                    gpts[name].quantizer.configure(
                        config.gptq.wbits, perchannel=True, sym=config.gptq.sym, mse=True, trits=config.gptq.trits
                    )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        calib_total = gpt_all['input'].shape[0]
        calib_report_interval = max(1, calib_total // self.calib_report_divisor)
        calib_start = time.time()
        for j in range(calib_total):
            inp = gpt_all['input'][j].unsqueeze(0).to(self.device)
            additional_layer_inputs = self._build_layer_inputs(1)
            _ = current_layer(inp, **additional_layer_inputs)
            if self.rank == 0 and (j + 1) % calib_report_interval == 0:
                report_gptq_calib(j + 1, calib_total, time.time() - calib_start)
        for h in handles:
            h.remove()

        gptq_losses = []
        linear_names = list(gpts.keys())
        linear_start = time.time()
        for li, name in enumerate(linear_names):
            if self.rank == 0:
                report_gptq_linear(li + 1, len(linear_names), name,
                                   time.time() - linear_start)
            if self.world_size > 1:
                gpts[name].sync_H(self.world_size)
            Q, scales = gpts[name].fasterquant(
                logging, percdamp=config.gptq.percdamp, blocksize=config.gptq.blocksize, groupsize=config.gptq.groupsize, static_groups=config.gptq.static_groups, prunen=config.gptq.prunen, prunem=config.gptq.prunem
            )
            if hasattr(gpts[name], 'last_gptq_loss'):
                gptq_losses.append(gpts[name].last_gptq_loss)
            if "q_proj" in name or "k_proj" in name or "in_proj_qkv" in name:
                self.update_quantized_weights(name, (Q, scales))
                if self.world_size > 1:
                    layer = self._get_layer_by_name(name)
                    dist.broadcast(layer.weight.data, src=0)
            else:
                trainer.setup_layer_training(name, Q, scales)
            gpts[name].free()

        if gptq_losses:
            trainer.gptq_avg_loss = sum(gptq_losses) / len(gptq_losses)

    def get_mlp_input(self, batch):
        current_layer = self.get_layer_module(self.current_layer_idx)
        additional_layer_inputs = self._build_layer_inputs(batch.shape[0])

        residual = batch
        hidden_states = current_layer.input_layernorm(batch)
        if current_layer.layer_type == "linear_attention":
            hidden_states = current_layer.linear_attn(hidden_states=hidden_states, attention_mask=additional_layer_inputs['attention_mask'])
        else:
            hidden_states, _ = current_layer.self_attn(hidden_states=hidden_states, **additional_layer_inputs)
        hidden_states = residual + hidden_states
        return hidden_states

    def get_mlp_output(self, mlp_input_batch):
        current_layer = self.get_layer_module(self.current_layer_idx)

        residual = mlp_input_batch
        hidden_states = current_layer.post_attention_layernorm(mlp_input_batch)
        hidden_states = current_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward_sharded_mlp(self, batch, quantized_weights):
        current_layer = self.get_layer_module(self.current_layer_idx)
        mlp = current_layer.mlp
        base = f"{self.get_current_layer()}.mlp"

        gate_w = quantized_weights[f"{base}.gate_proj"]
        up_w = quantized_weights[f"{base}.up_proj"]
        down_w = quantized_weights[f"{base}.down_proj"]

        mlp_input = self.get_mlp_input(batch)
        residual = mlp_input
        hidden_states = current_layer.post_attention_layernorm(mlp_input)

        if mlp.gate_proj.bias is not None or mlp.up_proj.bias is not None:
            raise NotImplementedError(
                "tensor_sharded Qwen3.5 MLP currently assumes bias-free "
                "gate_proj/up_proj"
            )

        gate_local = F.linear(hidden_states, gate_w, bias=None)
        up_local = F.linear(hidden_states, up_w, bias=None)
        hidden_local = mlp.act_fn(gate_local) * up_local
        partial = F.linear(hidden_local, down_w, bias=None)

        mlp_out = row_parallel_sum(partial)

        if mlp.down_proj.bias is not None:
            mlp_out = mlp_out + mlp.down_proj.bias

        return residual + mlp_out

    def _layer_prefixes(self, layer_name):
        layer_idx = int(layer_name.split(".")[-1])
        base = f"{self.layer_prefix}.{layer_idx}"
        layer = self.get_layer_module(layer_idx)

        if layer.layer_type == "linear_attention":
            non_mlp = [
                f"{base}.input_layernorm",
                f"{base}.linear_attn",
                f"{base}.post_attention_layernorm",
            ]
            mlp = [
                f"{base}.mlp"
            ]
        else:
            non_mlp = [
                f"{base}.input_layernorm",
                f"{base}.self_attn",
                f"{base}.post_attention_layernorm",
            ]
            mlp = [
                f"{base}.mlp"
            ]

        return {"non_mlp": non_mlp, "mlp": mlp}

    def move_embed_to(self, device):
        names = self._names_from_ckpt(["model.language_model.embed_tokens"])
        if device == "cuda":
            self._set_tensors(names)
            self.model.model.rotary_emb = self.model.model.rotary_emb.to("cuda")
        else:
            self._offload_names_to_meta(names)

    def move_output_heads_to(self, device):
        names = []
        names += self._names_from_ckpt("model.language_model.norm")
        names += self._names_from_ckpt("lm_head")
        if device == "cuda":
            self._set_tensors(names)
            if getattr(self.model.config, "tie_word_embeddings", False):
                self.model.tie_weights()
        else:
            self._offload_names_to_meta(names)

    def forward_with_quantized(self, batch, quantized_weights, self_attn):
        class LinearWeightHook:
            def __init__(self, module, quantized_weight):
                self.module = module
                self.quantized_weight = quantized_weight
                self.original_forward = module.forward

            def __enter__(self):
                def new_forward(module_self, x):
                    return torch.nn.functional.linear(x, self.quantized_weight, self.module.bias)
                self.module.forward = new_forward.__get__(self.module, torch.nn.Linear)

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.module.forward = self.original_forward

        current_layer = self.get_layer_module(self.current_layer_idx)
        hooks = []
        try:
            if quantized_weights is not None:
                if self_attn:
                    if current_layer.layer_type != "full_attention":
                        layer = current_layer.linear_attn
                    else:
                        layer = current_layer.self_attn
                else:
                    layer = current_layer.mlp

                for name, module in layer.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        if self_attn:
                            if current_layer.layer_type != "full_attention":
                                key = f"{self.get_current_layer()}.linear_attn.{name}"
                            else:
                                key = f"{self.get_current_layer()}.self_attn.{name}"
                        else: 
                            key = f"{self.get_current_layer()}.mlp.{name}"
                        if key in quantized_weights:
                            hooks.append(LinearWeightHook(module, quantized_weights[key]))
            else:
                if current_layer.layer_type != "full_attention":
                    for name, module in current_layer.linear_attn.named_modules():
                        key = f"{self.get_current_layer()}.linear_attn.{name}"
                        if key in self.temp_weights:
                            hooks.append(LinearWeightHook(module, self.temp_weights[key]))
                else:
                    for name, module in current_layer.self_attn.named_modules():
                        key = f"{self.get_current_layer()}.self_attn.{name}"
                        if key in self.temp_weights:
                            hooks.append(LinearWeightHook(module, self.temp_weights[key]))

            with ExitStack() as stack:
                for h in hooks: stack.enter_context(h)
                if self_attn:
                    output = self.get_mlp_input(batch)
                else:
                    additional_layer_inputs = self._build_layer_inputs(batch.shape[0])
                    output = current_layer(batch, **additional_layer_inputs)
            return output
        finally:
            hooks.clear()

    def save_attention_to_disc(self, pfx):
        os.makedirs(self.save_dir, exist_ok=True)
        to_save = {}
        current_layer = self.get_layer_module(self.current_layer_idx)
        if current_layer.layer_type == "linear_attention":
            pfx = pfx.replace("self_attn", "linear_attn")
            compressed_attn_layers = ['in_proj_qkv', 'in_proj_z', 'out_proj']
            non_compressed_attn_layers = ['conv1d', 'in_proj_a', 'in_proj_b', 'norm']
        else:
            compressed_attn_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
            non_compressed_attn_layers = ['q_norm', 'k_norm']
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
        if current_layer.layer_type == "linear_attention":
            to_save[f"{pfx}.A_log"] = current_layer.linear_attn.A_log.data
            to_save[f"{pfx}.dt_bias"] = current_layer.linear_attn.dt_bias.data
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
            rank=self.rank,
        )

        pad_token_id = getattr(self.model.config, "pad_token_id", None)
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

            for b in range(num_batches):
                ids_cpu = input_ids_cpu_list[b]
                x_cpu = activations_cpu_list[b]

                if x_cpu is None:
                    ids = ids_cpu.to(self.device, non_blocking=True)
                    x = self.model.model.embed_tokens(ids)
                else:
                    x = x_cpu.to(self.device, non_blocking=True)

                x = self.get_mlp_input(x)
                x = self.get_mlp_output(x)

                activations_cpu_list[b] = x.to("cpu", non_blocking=True).pin_memory()

            self.offload_to_meta(layer_name)
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

        nll_sum = torch.tensor(0.0, device=self.device)
        tok_cnt = torch.tensor(0.0, device=self.device)

        for b in range(num_batches):
            ids_cpu = input_ids_cpu_list[b]
            x_cpu = activations_cpu_list[b]

            input_ids = ids_cpu.to(self.device, non_blocking=True)
            x = x_cpu.to(self.device, non_blocking=True)

            x = self.model.model.norm(x)
            logits = self.model.lm_head(x)

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
