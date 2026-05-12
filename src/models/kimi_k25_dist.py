import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
import gc
from contextlib import ExitStack
from .kimi_k25 import KimiK25Wrapper
from src.moe.placement import ExpertSharder
from src.moe.autograd_ops import AllToAllTokens
from src.prior.gptq import *
from src.evaluation.wiki_eval import *
from src.utils.progress_reporter import report_gptq_calib, report_ppl_layer

class KimiK25DistributedWrapper(KimiK25Wrapper):
    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype, dummy=False):
        super().__init__(model_name, tokenizer, batch_size, seqlen, device, dtype, dummy=dummy)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.sharder = ExpertSharder(num_experts=self.num_experts, world_size=self.world_size)
        self.groupsize = 32

        self._owner_lut = torch.tensor(
            [self.sharder.owner(e) for e in range(self.num_experts)],
            dtype=torch.long
        )

    def _layer_prefixes(self, layer_name):
        layer_idx = int(layer_name.split('.')[-1])
        base = f"{self.layer_prefix}.{layer_idx}"
        if layer_idx < self.model.language_model.config.first_k_dense_replace:
            non_mlp = [
                f"{base}.input_layernorm",
                f"{base}.self_attn",
                f"{base}.post_attention_layernorm"
            ]
            local_expert = [
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
            local_expert = [
                f"{base}.mlp.experts.{e}"
                for e in range(self.num_experts)
                if self.sharder.owner(e) == self.rank
            ]
        return {"non_mlp": non_mlp, "mlp": local_expert}

    @torch.no_grad()
    def get_layer_activations(self, data_all):
        current_layer = self.get_layer_module(self.current_layer_idx)
        num_samples = data_all['input'].shape[0]
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        for batch_idx in range(num_batches):
            start_idx, end_idx = batch_idx * self.batch_size, min((batch_idx + 1) * self.batch_size, num_samples)
            x = data_all['input'][start_idx:end_idx].to(self.device, non_blocking=True)
            additional_layer_inputs = self._build_layer_inputs(x.shape[0])

            hidden_states = current_layer.input_layernorm(x)
            attn_out, _, _ = current_layer.self_attn(hidden_states, **additional_layer_inputs)
            mlp_input = x + attn_out
            if self.current_layer_idx < self.model.language_model.config.first_k_dense_replace:
                hidden_states = current_layer.post_attention_layernorm(mlp_input)
                hidden_states = current_layer.mlp(hidden_states)
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
                out = hidden_states + mlp_input
            else:
                out = self.run_expert_parallel(mlp_input)

            data_all['input'][start_idx:end_idx] = out.detach().cpu()      

    @torch.no_grad()
    def get_mlp_output(self, mlp_input_batch):
        return self.run_expert_parallel(mlp_input_batch)

    def _dispatch_tokens(self, mlp_input_batch):
        """Route tokens to expert-owning ranks via all-to-all.

        Returns (x_flat, hidden, send_idx_flat, in_split_sizes, out_split_sizes,
                 xin, win_or_none, eids, B, T, H).
        *win_or_none* is the routing-weight tensor when called from
        run_expert_parallel (topw provided) and None when called from
        calculate_mse (weights not needed).
        """
        layer = self.get_layer_module(self.current_layer_idx)
        device = self.device
        pg = dist.group.WORLD

        B, T, H = mlp_input_batch.shape
        hidden = layer.post_attention_layernorm(mlp_input_batch)
        x_flat = hidden.reshape(B * T, H)

        top_k = self.model.language_model.config.num_experts_per_tok
        topi, topw = layer.mlp.gate(hidden)

        tok_idx_flat = torch.arange(B * T, device=device, dtype=torch.long).repeat_interleave(top_k)
        eid_flat = topi.reshape(-1).to(torch.long)
        w_flat = topw.reshape(-1).to(self.dtype)

        owner_lut = self._owner_lut.to(device)
        owners_flat = owner_lut[eid_flat]
        perm = torch.argsort(owners_flat, stable=True)
        owners_flat = owners_flat.index_select(0, perm)
        send_idx_flat = tok_idx_flat.index_select(0, perm)
        send_eid_flat = eid_flat.index_select(0, perm)
        send_w_flat = w_flat.index_select(0, perm)
        send_x_flat = x_flat.index_select(0, send_idx_flat)

        world_size = self.world_size
        in_sizes_tensor = torch.bincount(owners_flat, minlength=world_size).to(torch.long)
        all_sizes = [torch.empty_like(in_sizes_tensor) for _ in range(world_size)]
        dist.all_gather(all_sizes, in_sizes_tensor, group=pg)
        recv_sizes = torch.stack(all_sizes)[:, self.rank]
        out_split_sizes = recv_sizes.tolist()
        in_split_sizes = in_sizes_tensor.tolist()

        xin = AllToAllTokens.apply(send_x_flat, out_split_sizes, in_split_sizes, pg)
        win = AllToAllTokens.apply(send_w_flat.unsqueeze(1), out_split_sizes, in_split_sizes, pg).to(self.dtype)
        eids = AllToAllTokens.apply(send_eid_flat.unsqueeze(1), out_split_sizes, in_split_sizes, pg).squeeze(1)

        return x_flat, hidden, send_idx_flat, in_split_sizes, out_split_sizes, xin, win, eids, B, T, H

    def _batched_expert_forward(self, xin, eids, quantized_weights=None):
        """Process all local experts in a single batched pass.

        Tokens in *xin* are grouped by expert ID.  For each expert the MLP is
        gate_proj -> silu, up_proj -> *, down_proj. When *quantized_weights* is
        given the quantized weight matrices are used instead of the module
        weights.
        """
        layer = self.get_layer_module(self.current_layer_idx)
        layer_key = self.get_current_layer()

        eids_long = eids.to(torch.long)
        unique_eids, inverse, counts = torch.unique(eids_long, sorted=True, return_inverse=True, return_counts=True)

        sort_idx = torch.argsort(inverse, stable=True)
        sorted_x = xin.index_select(0, sort_idx)

        out_buf = torch.empty_like(sorted_x)

        offset = 0
        for i, eid_val in enumerate(unique_eids.tolist()):
            n = counts[i].item()
            inp_e = sorted_x[offset:offset + n]
            expert = layer.mlp.experts[eid_val]

            if quantized_weights is not None:
                key = f"{layer_key}.mlp.experts.{eid_val}"
                qw = quantized_weights[key]
                gate_out = F.linear(inp_e, qw["gate_proj"][0] if isinstance(qw["gate_proj"], tuple) else qw["gate_proj"])
                up_out = F.linear(inp_e, qw["up_proj"][0] if isinstance(qw["up_proj"], tuple) else qw["up_proj"])
                hidden = torch.nn.functional.silu(gate_out) * up_out
                out_e = F.linear(hidden, qw["down_proj"][0] if isinstance(qw["down_proj"], tuple) else qw["down_proj"])
            else:
                out_e = expert(inp_e)

            out_buf[offset:offset + n] = out_e
            offset += n

        unsort_idx = torch.argsort(sort_idx)
        return out_buf.index_select(0, unsort_idx)

    def run_expert_parallel(self, mlp_input_batch, quantized_weights=None):
        pg = dist.group.WORLD

        x_flat, hidden, send_idx_flat, in_split_sizes, out_split_sizes, xin, win, eids, B, T, H = \
            self._dispatch_tokens(mlp_input_batch)

        out_local = self._batched_expert_forward(xin, eids, quantized_weights)
        xin = out_local * win

        returned = AllToAllTokens.apply(xin, in_split_sizes, out_split_sizes, pg)

        y_flat = x_flat.new_zeros(x_flat.shape)
        y_flat.index_add_(0, send_idx_flat, returned)
        y = y_flat.view(B, T, H)

        layer = self.get_layer_module(self.current_layer_idx)
        y = y + layer.mlp.shared_experts(hidden)

        return y + mlp_input_batch

    def _forward_single_expert(self, layer, expert_id, x_e, quantized_weights):
        expert = layer.mlp.experts[expert_id]

        class LinearWeightHook:
            def __init__(self, module, qweight):
                self.module = module
                self.qw = qweight
                self.saved = module.forward
            def __enter__(self):
                def new_forward(module_self, x):
                    return F.linear(x, self.qw, self.module.bias)
                self.module.forward = new_forward.__get__(self.module, torch.nn.Linear)
            def __exit__(self, a,b,c):
                self.module.forward = self.saved

        hooks = []
        try:
            if quantized_weights is not None:
                for name, module in expert.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        key = f"{self.get_current_layer()}.mlp.experts.{expert_id}"
                        hooks.append(LinearWeightHook(module, quantized_weights[key][name]))
            with ExitStack() as stack:
                for h in hooks: stack.enter_context(h)
                out = expert(x_e)
            return out
        finally:
            hooks.clear()

    def calculate_mse(self, mlp_input_batch, quantized_weights, self_attn=False, validation=False, accumulation_steps=1):
        layer = self.get_layer_module(self.current_layer_idx)
        device = self.device
        pg = dist.group.WORLD

        B, T, H = mlp_input_batch.shape
        with torch.no_grad():
            hidden = layer.post_attention_layernorm(mlp_input_batch)
            x_flat = hidden.reshape(B * T, H)

            top_k = self.model.language_model.config.num_experts_per_tok
            topi, _ = layer.mlp.gate(hidden)

            tok_idx_flat = torch.arange(B * T, device=device, dtype=torch.long).repeat_interleave(top_k)
            eid_flat = topi.reshape(-1).to(torch.long)

            owner_lut = self._owner_lut.to(device)
            owners_flat = owner_lut[eid_flat]
            perm = torch.argsort(owners_flat, stable=True)
            owners_flat = owners_flat.index_select(0, perm)
            send_idx_flat = tok_idx_flat.index_select(0, perm)
            send_eid_flat = eid_flat.index_select(0, perm)
            send_x_flat = x_flat.index_select(0, send_idx_flat)

            world_size = self.world_size
            in_sizes_tensor = torch.bincount(owners_flat, minlength=world_size).to(torch.long)
            all_sizes = [torch.empty_like(in_sizes_tensor) for _ in range(world_size)]
            dist.all_gather(all_sizes, in_sizes_tensor, group=pg)
            recv_sizes = torch.stack(all_sizes)[:, self.rank]
            out_split_sizes = recv_sizes.tolist()
            in_split_sizes = in_sizes_tensor.tolist()

            xin = AllToAllTokens.apply(send_x_flat, out_split_sizes, in_split_sizes, pg)
            eids = AllToAllTokens.apply(send_eid_flat.unsqueeze(1), out_split_sizes, in_split_sizes, pg).squeeze(1)

        with torch.no_grad():
            out_fp = self._batched_expert_forward(xin, eids, quantized_weights=None)
        out_q = self._batched_expert_forward(xin, eids, quantized_weights=quantized_weights)

        total_mse = self.loss_fn(out_q, out_fp)
        if not validation:
            (total_mse / accumulation_steps).backward()

        return total_mse.item()

    def get_layer_initialization(self, trainer, gpt_all, config, logging):
        if logging is not None:
            logging = logging.logger
        layer_idx = self.current_layer_idx
        layer = self.get_layer_module(layer_idx)
        rank = self.rank
        self.groupsize = config.quantization.groupsize

        owned_experts = [e for e in range(self.num_experts) if self.sharder.owner(e) == rank]
        subset = {}

        for e in owned_experts:
            expert = layer.mlp.experts[e]
            base_prefix = f"{self.get_current_layer()}.mlp.experts.{e}"
            for name, module in expert.named_modules():
                if isinstance(module, torch.nn.Linear):
                    subset[f"{base_prefix}.{name}"] = module

        init_method = config.quantization.init_method
        gsq_enabled = config.quantization.gsq_enabled

        if init_method in ("rtn", "random"):
            quantize_fn = random_quantize if init_method == "random" else rtn_quantize
            for name in subset:
                Q, scales = quantize_fn(subset[name], config, self.device, self.dtype)
                if gsq_enabled:
                    trainer.setup_layer_training(name, Q, scales)
                else:
                    self.update_quantized_weights(name, (Q, scales))
            dist.barrier()
            return

        if init_method != "gptq":
            raise ValueError(
                f"Unknown init_method={init_method!r}. Supported: 'gptq', 'rtn', 'random'"
            )

        gpts = {}
        for name in subset:
            gpts[name] = GPTQ(subset[name], name, config, self.device, self.dtype)
            if config.gptq.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    config.gptq.wbits, perchannel=True, sym=config.gptq.sym, mse=True, trits=config.gptq.trits
                )

        def _add_batch(full_key):
            def _hook(_, inp, out):
                gpts[full_key].add_batch(inp[0].data, out.data)
            return _hook

        handles = []
        n_hessian = config.gptq.nsamples // self.world_size
        try:
            for full_name, module in subset.items():
                if "up_proj" in full_name:
                    continue
                handles.append(module.register_forward_hook(_add_batch(full_name)))

            if logging is not None and rank == 0:
                logging.info(f"GPTQ Hessian accumulation: {n_hessian} samples, {len(owned_experts)} local experts")
            calib_start = time.time()
            calib_report_interval = max(1, n_hessian // self.calib_report_divisor)
            with torch.no_grad():
                for j in range(n_hessian):
                    x = gpt_all['input'][j].unsqueeze(0).to(self.device, non_blocking=True)
    
                    additional_layer_inputs = self._build_layer_inputs(x.shape[0])
    
                    hidden_states = layer.input_layernorm(x)
                    attn_out, _, _ = layer.self_attn(hidden_states, **additional_layer_inputs)
                    x = x + attn_out
    
                    _ = self.run_expert_parallel(x, quantized_weights=None)
                    if rank == 0 and (j + 1) % calib_report_interval == 0:
                        report_gptq_calib(j + 1, n_hessian, time.time() - calib_start)
        finally:
            for h in handles:
                h.remove()
                
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        gptq_losses = []
        for name in gpts:
            if "up_proj" in name:
                continue
            Q, scales = gpts[name].fasterquant(
                logging,
                percdamp=config.gptq.percdamp,
                blocksize=config.gptq.blocksize,
                groupsize=config.gptq.groupsize,
                static_groups=config.gptq.static_groups,
                prunen=config.gptq.prunen, 
                prunem=config.gptq.prunem
            )
            if hasattr(gpts[name], 'last_gptq_loss'):
                gptq_losses.append(gpts[name].last_gptq_loss)
            if scales is not None and gsq_enabled:
                trainer.setup_layer_training(name, Q, scales)
            else:
                self.update_quantized_weights(name, (Q, scales) if scales is not None else Q)
            if "gate_proj" in name:
                base = name[: -len(".gate_proj")]
                new_name = f"{base}.up_proj"
                gpts[new_name].H = gpts[name].H
                gpts[new_name].dead = gpts[name].dead
                Q, scales = gpts[new_name].fasterquant(
                    logging,
                    percdamp=config.gptq.percdamp,
                    blocksize=config.gptq.blocksize,
                    groupsize=config.gptq.groupsize,
                    static_groups=config.gptq.static_groups,
                    calculate_cholesky=False,
                    prunen=config.gptq.prunen, 
                    prunem=config.gptq.prunem
                )
                if hasattr(gpts[new_name], 'last_gptq_loss'):
                    gptq_losses.append(gpts[new_name].last_gptq_loss)
                if scales is not None and gsq_enabled:
                    trainer.setup_layer_training(new_name, Q, scales)
                else:
                    self.update_quantized_weights(new_name, (Q, scales) if scales is not None else Q)
                gpts[name].free()
                gpts[new_name].free()
            else:
                gpts[name].free()

        if gptq_losses:
            trainer.gptq_avg_loss = sum(gptq_losses) / len(gptq_losses)

        dist.barrier()

    def _load_layer_for_eval(self, layer_idx, read_from_disk):
        layer_name = f"{self.layer_prefix}.{layer_idx}"
        if layer_idx <= read_from_disk and layer_idx >= self.model.language_model.config.first_k_dense_replace:
            self.load_from_disc(layer_name)
        else:
            self.move_layer_to_gpu(layer_name)

    @torch.no_grad()
    def ppl_evaluation(self, read_from_disk=-1):
        dataset = get_dataset("open_thoughts", self.tokenizer)
        testloader = prepare_test_dataloader(
                dataset=dataset["test"], 
                tokenizer=self.tokenizer, 
                seqlen=self.model.seqlen,
                batch_size=4,
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

        transfer_stream = torch.cuda.Stream(device=self.device)

        self.current_layer_idx = 0
        self._load_layer_for_eval(0, read_from_disk)
        ppl_start = time.time()

        for i in range(self.num_layers):
            if self.rank == 0:
                report_ppl_layer(i, self.num_layers, elapsed=time.time() - ppl_start)
            self.current_layer_idx = i
            layer_name = f"{self.layer_prefix}.{i}"
            layer = self.get_layer_module(i)

            for b in range(num_batches):
                ids_cpu = input_ids_cpu_list[b]

                x_cpu = activations_cpu_list[b]
                if x_cpu is None:
                    ids = ids_cpu.to(self.device, non_blocking=True)
                    x = self.model.language_model.model.embed_tokens(ids)
                else:
                    x = x_cpu.to(self.device, non_blocking=True)

                additional_layer_inputs = self._build_layer_inputs(x.shape[0])
                hidden = layer.input_layernorm(x)
                attn_out, _, _ = layer.self_attn(hidden, **additional_layer_inputs)
                x = x + attn_out
                if i < self.model.language_model.config.first_k_dense_replace:
                    hidden_states = layer.post_attention_layernorm(x)
                    hidden_states = layer.mlp(hidden_states)
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[0]
                    x = hidden_states + x
                else:
                    x = self.run_expert_parallel(x)

                activations_cpu_list[b] = x.to("cpu", non_blocking=True).pin_memory()

            self.offload_to_meta(layer_name)
            torch.cuda.synchronize(self.device)

            if i + 1 < self.num_layers:
                with torch.cuda.stream(transfer_stream):
                    self._load_layer_for_eval(i + 1, read_from_disk)
                transfer_stream.synchronize()

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

        dist.all_reduce(local_nll_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_tok_cnt, op=dist.ReduceOp.SUM)

        mean_nll = (local_nll_sum / local_tok_cnt).item()
        ppl = math.exp(mean_nll)

        torch.cuda.synchronize(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        return ppl
