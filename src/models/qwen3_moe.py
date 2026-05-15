import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gc
from .base import BaseModelWrapper
from src.evaluation.wiki_eval import *
from src.utils.progress_reporter import report_gptq_calib, report_gptq_linear
from src.prior.gptq import GPTQ, rtn_quantize
from src.prior.quant import Quantizer


class Qwen3MoeWrapper(BaseModelWrapper):
    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype):
        super().__init__(model_name, tokenizer, batch_size, seqlen, device, dtype)
        self.layer_prefix = "model.layers"
        self.num_layers = len(self.model.model.layers)
        self.num_experts = self.model.config.num_experts
        self.decoder_sparse_step = getattr(self.model.config, 'decoder_sparse_step', 1)
        self.mlp_only_layers = getattr(self.model.config, 'mlp_only_layers', [])
        self.is_moe = True
        self.fused_experts = True
        self.fused_expert_intermediate_size = getattr(
            self.model.config, "moe_intermediate_size", None
        ) or getattr(self.model.config, "intermediate_size", None)

    def _is_moe_layer(self, layer_idx):
        if layer_idx in self.mlp_only_layers:
            return False
        return self.num_experts > 0 and (layer_idx + 1) % self.decoder_sparse_step == 0

    def _layer_prefixes(self, layer_name):
        layer_idx = int(layer_name.split('.')[-1])
        base = f"{self.layer_prefix}.{layer_idx}"
        if not self._is_moe_layer(layer_idx):
            non_mlp = [
                f"{base}.input_layernorm",
                f"{base}.self_attn",
                f"{base}.post_attention_layernorm"
            ]
            mlp = [
                f"{base}.mlp"
            ]
            return {"non_mlp": non_mlp, "mlp": mlp}
        non_mlp = [
            f"{base}.input_layernorm",
            f"{base}.self_attn",
            f"{base}.mlp.gate",
            f"{base}.post_attention_layernorm"
        ]
        mlp = [
            f"{base}.mlp.experts.{e}"
            for e in range(self.num_experts)
        ]
        mlp_offload_params = [
            f"{base}.mlp.experts.gate_up_proj",
            f"{base}.mlp.experts.down_proj",
        ]
        return {"non_mlp": non_mlp, "mlp": mlp, "mlp_offload_params": mlp_offload_params}

    def _virtual_expert_linear(self, layer_idx, expert_id, proj_name):
        layer = self.model.model.layers[layer_idx]
        experts = layer.mlp.experts
        intermediate_dim = self.fused_expert_intermediate_size
        hidden_dim = self.model.config.hidden_size
        if proj_name == "gate_proj":
            W = experts.gate_up_proj.data[expert_id, :intermediate_dim, :]
            lin = nn.Linear(hidden_dim, intermediate_dim, bias=False, device=W.device, dtype=W.dtype)
            lin.weight = nn.Parameter(W, requires_grad=False)
        elif proj_name == "up_proj":
            W = experts.gate_up_proj.data[expert_id, intermediate_dim : 2 * intermediate_dim, :]
            lin = nn.Linear(hidden_dim, intermediate_dim, bias=False, device=W.device, dtype=W.dtype)
            lin.weight = nn.Parameter(W, requires_grad=False)
        else:
            W = experts.down_proj.data[expert_id, :, :]
            lin = nn.Linear(intermediate_dim, hidden_dim, bias=False, device=W.device, dtype=W.dtype)
            lin.weight = nn.Parameter(W, requires_grad=False)
        return lin

    def _fused_expert_forward_batched(self, xin, eids, quantized_weights=None, gpts_calib=None):
        layer = self.get_layer_module(self.current_layer_idx)
        layer_key = self.get_current_layer()
        experts = layer.mlp.experts
        intermediate_dim = self.fused_expert_intermediate_size

        eids_long = eids.to(torch.long)
        unique_eids, inverse, counts = torch.unique(eids_long, sorted=True, return_inverse=True, return_counts=True)

        sort_idx = torch.argsort(inverse, stable=True)
        sorted_x = xin.index_select(0, sort_idx)

        out_buf = torch.empty_like(sorted_x)

        offset = 0
        for i, eid_val in enumerate(unique_eids.tolist()):
            n = counts[i].item()
            inp_e = sorted_x[offset:offset + n]
            gate_w = experts.gate_up_proj[eid_val, :intermediate_dim, :]
            up_w = experts.gate_up_proj[eid_val, intermediate_dim : 2 * intermediate_dim, :]
            down_w = experts.down_proj[eid_val, :, :]

            gate_out = F.linear(inp_e, gate_w)
            up_out = F.linear(inp_e, up_w)
            hidden_act = experts.act_fn(gate_out) * up_out
            out_e = F.linear(hidden_act, down_w)

            if quantized_weights is not None:
                key = f"{layer_key}.mlp.experts.{eid_val}"
                qw = quantized_weights[key]
                gate_out = F.linear(inp_e, qw["gate_proj"][0] if isinstance(qw["gate_proj"], tuple) else qw["gate_proj"])
                up_out = F.linear(inp_e, qw["up_proj"][0] if isinstance(qw["up_proj"], tuple) else qw["up_proj"])
                hidden_act = experts.act_fn(gate_out) * up_out
                out_e = F.linear(hidden_act, qw["down_proj"][0] if isinstance(qw["down_proj"], tuple) else qw["down_proj"])

            if gpts_calib is not None:
                lk = f"{layer_key}.mlp.experts.{eid_val}"
                gk = f"{lk}.gate_proj"
                dk = f"{lk}.down_proj"
                if gk in gpts_calib:
                    gpts_calib[gk].add_batch(inp_e.data, gate_out.data)
                if dk in gpts_calib:
                    gpts_calib[dk].add_batch(hidden_act.data, out_e.data)

            out_buf[offset:offset + n] = out_e
            offset += n

        unsort_idx = torch.argsort(sort_idx)
        return out_buf.index_select(0, unsort_idx)

    def _route_tokens_flat(self, mlp_input_batch):
        layer = self.get_layer_module(self.current_layer_idx)
        B, T, H = mlp_input_batch.shape
        hidden = layer.post_attention_layernorm(mlp_input_batch)
        x_flat = hidden.reshape(B * T, H)
        router = layer.mlp.gate
        _, topw, topi = router(hidden.reshape(-1, H))
        top_k = router.top_k
        tok_idx_flat = torch.arange(B * T, device=self.device, dtype=torch.long).repeat_interleave(top_k)
        eid_flat = topi.reshape(-1).to(torch.long)
        w_flat = topw.reshape(-1).to(self.dtype)
        return x_flat, tok_idx_flat, eid_flat, w_flat, hidden

    def _local_sparse_moe(self, mlp_input_batch, quantized_weights=None, gpts_calib=None):
        layer = self.get_layer_module(self.current_layer_idx)
        B, T, H = mlp_input_batch.shape
        x_flat, tok_idx_flat, eid_flat, w_flat, hidden = self._route_tokens_flat(mlp_input_batch)
        router = layer.mlp.gate
        top_k = router.top_k

        sort_perm = torch.argsort(eid_flat, stable=True)
        x_sorted = x_flat.index_select(0, tok_idx_flat).index_select(0, sort_perm)
        e_sorted = eid_flat.index_select(0, sort_perm)
        w_sorted = w_flat.index_select(0, sort_perm)

        splits = torch.bincount(e_sorted, minlength=self.num_experts)
        expert_chunks = torch.split(x_sorted, splits.tolist())
        weight_chunks = torch.split(w_sorted, splits.tolist())
        eid_chunks = torch.split(e_sorted, splits.tolist())

        outs = []
        idx_run = 0
        for eid in range(self.num_experts):
            n_e = splits[eid].item()
            if n_e == 0:
                continue
            inp_chunk = expert_chunks[eid]
            wei_chunk = weight_chunks[eid]
            out_chunk = self._fused_expert_forward_batched(
                inp_chunk, torch.full((n_e,), eid, device=self.device, dtype=torch.long),
                quantized_weights=quantized_weights, gpts_calib=gpts_calib)
            outs.append(out_chunk * wei_chunk.unsqueeze(1))
            idx_run += n_e

        if not outs:
            return mlp_input_batch

        out_gather = torch.cat(outs, dim=0)
        inv_perm = torch.argsort(sort_perm)
        out_gather = out_gather.index_select(0, inv_perm)

        y_flat = x_flat.new_zeros(x_flat.shape)
        y_flat.index_add_(0, tok_idx_flat, out_gather)
        return y_flat.view(B, T, H) + mlp_input_batch

    def move_embed_to(self, device):
        names = self._names_from_ckpt(["model.embed_tokens"])
        if device == "cuda":
            self._set_tensors(names)
        else:
            self._offload_names_to_meta(names)

    def move_output_heads_to(self, device):
        names = []
        names += self._names_from_ckpt("model.norm")
        names += self._names_from_ckpt("lm_head")
        if device == "cuda":
            self._set_tensors(names)
        else:
            self._offload_names_to_meta(names)

    def get_mlp_input(self, batch):
        current_layer = self.get_layer_module(self.current_layer_idx)

        additional_layer_inputs = {"attention_mask": None}
        for k, v in self.kwargs.items():
            additional_layer_inputs[k] = v

        hidden_states = current_layer.input_layernorm(batch)
        hidden_states, _ = current_layer.self_attn(hidden_states, **additional_layer_inputs)
        return hidden_states + batch

    def get_mlp_output(self, mlp_input_batch):
        current_layer = self.get_layer_module(self.current_layer_idx)

        if not self._is_moe_layer(self.current_layer_idx):
            hidden_states = current_layer.post_attention_layernorm(mlp_input_batch)
            hidden_states = current_layer.mlp(hidden_states)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            return hidden_states + mlp_input_batch
        return self._local_sparse_moe(mlp_input_batch)

    def get_layer_module(self, idx):
        return self.model.model.layers[idx]

    def update_quantized_weights(self, layer_name, quantized_weights):
        ln = layer_name.replace(".language_model", "")
        parsed = self._parse_fused_expert_module_base(ln)
        if parsed and self.fused_experts:
            layer_prefix, expert_idx, proj_kind = parsed
            experts_mod = self.model.get_submodule(f"{layer_prefix}.mlp.experts")
            intermediate_dim = self.fused_expert_intermediate_size
            if isinstance(quantized_weights, tuple):
                Q, scales = quantized_weights
                Q = Q.to(self.device).to(experts_mod.gate_up_proj.dtype)
                self.temp_weights[layer_name] = Q
                self.temp_weights[f"{layer_name}.scale"] = scales
                with torch.no_grad():
                    if proj_kind == "gate_proj":
                        experts_mod.gate_up_proj.data[expert_idx, :intermediate_dim, :].copy_(Q)
                    elif proj_kind == "up_proj":
                        experts_mod.gate_up_proj.data[expert_idx, intermediate_dim : 2 * intermediate_dim, :].copy_(Q)
                    else:
                        experts_mod.down_proj.data[expert_idx, :, :].copy_(Q)
            else:
                Q = quantized_weights.to(self.device).to(experts_mod.gate_up_proj.dtype)
                self.temp_weights[layer_name] = Q
                with torch.no_grad():
                    if proj_kind == "gate_proj":
                        experts_mod.gate_up_proj.data[expert_idx, :intermediate_dim, :].copy_(Q)
                    elif proj_kind == "up_proj":
                        experts_mod.gate_up_proj.data[expert_idx, intermediate_dim : 2 * intermediate_dim, :].copy_(Q)
                    else:
                        experts_mod.down_proj.data[expert_idx, :, :].copy_(Q)
            return
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

    def calculate_mse(self, batch, quantized_weights, self_attn=False, validation=False, accumulation_steps=1):
        if self_attn:
            return super().calculate_mse(batch, quantized_weights, self_attn=True,
                                         validation=validation, accumulation_steps=accumulation_steps)
        if not self._is_moe_layer(self.current_layer_idx):
            return super().calculate_mse(batch, quantized_weights, self_attn=False,
                                         validation=validation, accumulation_steps=accumulation_steps)
        with torch.no_grad():
            out_fp = self._local_sparse_moe(batch, quantized_weights=None)
        out_q = self._local_sparse_moe(batch, quantized_weights=quantized_weights)
        mse = self.loss_fn(out_q, out_fp)
        if not validation:
            (mse / accumulation_steps).backward()
        return mse.item()

    def get_layer_initialization(self, trainer, gpt_all, config, logging, is_attn=False):
        if is_attn:
            return super().get_layer_initialization(trainer, gpt_all, config, logging, is_attn=True)
        current_layer = self.get_layer_module(self.current_layer_idx)
        self.groupsize = config.quantization.groupsize
        if not self._is_moe_layer(self.current_layer_idx):
            return super().get_layer_initialization(trainer, gpt_all, config, logging, is_attn=False)

        if logging is not None:
            logging = logging.logger
        base_prefix = f"{self.get_current_layer()}.mlp"
        subset = {}
        for e in range(self.num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                subset[f"{base_prefix}.experts.{e}.{proj}"] = self._virtual_expert_linear(
                    self.current_layer_idx, e, proj)

        skip_gptq = getattr(config.gptq, 'skip', False)

        if skip_gptq:
            for name in subset:
                Q, scales = rtn_quantize(subset[name], config, self.device, self.dtype)
                trainer.setup_layer_training(name, Q, scales)
            return

        gpts = {}
        for name in subset:
            gpts[name] = GPTQ(subset[name], name, config, self.device, self.dtype)
            if config.gptq.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    config.gptq.wbits, perchannel=True, sym=config.gptq.sym, mse=True, trits=config.gptq.trits
                )

        calib_total = gpt_all['input'].shape[0]
        calib_report_interval = max(1, calib_total // self.calib_report_divisor)
        calib_start = time.time()
        for j in range(calib_total):
            inp = gpt_all['input'][j].unsqueeze(0).to(self.device)
            additional_layer_inputs = self._build_layer_inputs(1)
            hidden_states = current_layer.input_layernorm(inp)
            attn_out, _ = current_layer.self_attn(hidden_states, **additional_layer_inputs)
            x = inp + attn_out
            self._local_sparse_moe(x, quantized_weights=None, gpts_calib=gpts)
            if self.rank == 0 and (j + 1) % calib_report_interval == 0:
                report_gptq_calib(j + 1, calib_total, time.time() - calib_start)

        gptq_losses = []
        linear_names = list(gpts.keys())
        linear_start = time.time()
        for li, name in enumerate(linear_names):
            if self.rank == 0:
                report_gptq_linear(li + 1, len(linear_names), name,
                                   time.time() - linear_start)
            if "up_proj" in name:
                continue
            if self.world_size > 1:
                gpts[name].sync_H(self.world_size)
            Q, scales = gpts[name].fasterquant(
                logging, percdamp=config.gptq.percdamp, blocksize=config.gptq.blocksize, groupsize=config.gptq.groupsize, static_groups=config.gptq.static_groups, prunen=config.gptq.prunen, prunem=config.gptq.prunem
            )
            if hasattr(gpts[name], 'last_gptq_loss'):
                gptq_losses.append(gpts[name].last_gptq_loss)
            trainer.setup_layer_training(name, Q, scales)
            if "gate_proj" in name:
                base = name[: -len(".gate_proj")]
                new_name = f"{base}.up_proj"
                gpts[new_name].H = gpts[name].H
                gpts[new_name].dead = gpts[name].dead
                Q, scales = gpts[new_name].fasterquant(
                    logging,
                    percdamp=config.gptq.percdamp, blocksize=config.gptq.blocksize, groupsize=config.gptq.groupsize, static_groups=config.gptq.static_groups, calculate_cholesky=False, prunen=config.gptq.prunen, prunem=config.gptq.prunem
                )
                if hasattr(gpts[new_name], 'last_gptq_loss'):
                    gptq_losses.append(gpts[new_name].last_gptq_loss)
                trainer.setup_layer_training(new_name, Q, scales)
                gpts[name].free()
                gpts[new_name].free()
            else:
                gpts[name].free()

        if gptq_losses:
            trainer.gptq_avg_loss = sum(gptq_losses) / len(gptq_losses)

    def save_moe_experts_to_disc(self):
        if not self._is_moe_layer(self.current_layer_idx):
            return
        layer_key = self.get_current_layer()
        for e in range(self.num_experts):
            base = f"{layer_key}.mlp.experts.{e}"
            gk = f"{base}.gate_proj"
            if gk not in self.temp_weights:
                continue
            pairs = {
                "gate_proj": (self.temp_weights[gk], self.temp_weights[f"{gk}.scale"]),
                "up_proj": (
                    self.temp_weights[f"{base}.up_proj"],
                    self.temp_weights[f"{base}.up_proj.scale"],
                ),
                "down_proj": (
                    self.temp_weights[f"{base}.down_proj"],
                    self.temp_weights[f"{base}.down_proj.scale"],
                ),
            }
            self.save_to_disc(base, pairs)

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
            if i <= read_from_disk and self._is_moe_layer(i):
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

                hidden = layer.input_layernorm(x)
                attn_out, _ = layer.self_attn(hidden, **additional_layer_inputs)
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

            x = self.model.model.norm(x)
            logits = self.model.lm_head(x)

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
