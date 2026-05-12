import os
import gc
import time
import math
from contextlib import ExitStack

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
from accelerate.utils import set_module_tensor_to_device

from .qwen35_moe import Qwen35MoeWrapper
from src.moe.placement import ExpertSharder
from src.moe.autograd_ops import AllToAllTokens
from src.prior.gptq import *
from src.evaluation.wiki_eval import *
from src.utils.progress_reporter import report_gptq_calib, report_ppl_layer


class _VirtualLinear(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = None


class Qwen35MoeDistributedWrapper(Qwen35MoeWrapper):
    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype):
        super().__init__(model_name, tokenizer, batch_size, seqlen, device, dtype)

        if not dist.is_initialized():
            raise RuntimeError("Qwen35MoeDistributedWrapper requires torch.distributed to be initialized.")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.sharder = ExpertSharder(num_experts=self.num_experts, world_size=self.world_size)

        self.local_expert_ids = [
            e for e in range(self.num_experts)
            if self.sharder.owner(e) == self.rank
        ]

        self._owner_lut = torch.tensor(
            [self.sharder.owner(e) for e in range(self.num_experts)],
            dtype=torch.long,
            device=self.device
        )

        self._global_to_local_lut = torch.full(
            (self.num_experts,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        for local_pos, global_eid in enumerate(self.local_expert_ids):
            self._global_to_local_lut[global_eid] = local_pos

        self.groupsize = 32
        self.is_moe = True

    def _resolve_ckpt_tensor(self, canonical_name):
        candidates = [
            canonical_name,
            canonical_name.replace(".language_model", ""),
        ]

        if self._name_to_shard is not None:
            for c in candidates:
                if c in self._name_to_shard:
                    return os.path.join(self.ckpt_path, self._name_to_shard[c]), c
        else:
            with safe_open(self._single_shard, framework="pt", device="cpu") as f:
                keys = set(f.keys())
                for c in candidates:
                    if c in keys:
                        return self._single_shard, c

        raise KeyError(f"Could not find tensor {canonical_name} in checkpoint.")

    def _set_tensor_allow_shape_change(self, name, value):
        parent_name, leaf = name.replace(".language_model", "").rsplit(".", 1)
        parent = self.model.get_submodule(parent_name)

        old = getattr(parent, leaf)

        if isinstance(old, nn.Parameter):
            parent._parameters[leaf] = nn.Parameter(
                value,
                requires_grad=old.requires_grad,
            )
        elif leaf in parent._buffers:
            parent._buffers[leaf] = value
        else:
            setattr(parent, leaf, value)

    def _set_tensor_to_meta_allow_shape_change(self, name):
        parent_name, leaf = name.replace(".language_model", "").rsplit(".", 1)
        parent = self.model.get_submodule(parent_name)

        old = getattr(parent, leaf)
        meta_value = torch.empty(
            tuple(old.shape),
            dtype=old.dtype,
            device="meta",
        )

        if isinstance(old, nn.Parameter):
            parent._parameters[leaf] = nn.Parameter(
                meta_value,
                requires_grad=old.requires_grad,
            )
        elif leaf in parent._buffers:
            parent._buffers[leaf] = meta_value
        else:
            setattr(parent, leaf, meta_value)

    def _layer_prefixes(self, layer_name):
        layer_idx = int(layer_name.split(".")[-1])
        base = f"{self.layer_prefix}.{layer_idx}"

        attn_prefix = (
            f"{base}.linear_attn"
            if self._is_linear_attention_layer(layer_idx)
            else f"{base}.self_attn"
        )

        non_mlp = [
            f"{base}.input_layernorm",
            attn_prefix,
            f"{base}.mlp.gate",
            f"{base}.mlp.shared_expert",
            f"{base}.mlp.shared_expert_gate",
            f"{base}.post_attention_layernorm"
        ]

        mlp = [
            f"{base}.mlp.experts.{e}"
            for e in self.local_expert_ids
        ]

        return {"non_mlp": non_mlp, "mlp": mlp}

    def _expert_tensor_names(self, layer_idx):
        base = f"{self.layer_prefix}.{layer_idx}.mlp.experts"
        return [
            f"{base}.gate_up_proj",
            f"{base}.down_proj"
        ]

    @staticmethod
    def _contiguous_ranges(sorted_indices):
        if not sorted_indices:
            return []

        ranges = []
        start = sorted_indices[0]
        prev = sorted_indices[0]

        for x in sorted_indices[1:]:
            if x == prev + 1:
                prev = x
            else:
                ranges.append((start, prev + 1))
                start = x
                prev = x

        ranges.append((start, prev + 1))
        return ranges

    def _read_first_dim_indices(self, shard_path, tensor_name, indices):
        indices = list(map(int, indices))
        sorted_indices = indices

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            try:
                tensor_slice = f.get_slice(tensor_name)
                chunks = [
                    tensor_slice[start:end]
                    for start, end in self._contiguous_ranges(sorted_indices)
                ]
                return torch.cat(chunks, dim=0)
            except Exception:
                full = f.get_tensor(tensor_name)
                idx = torch.tensor(sorted_indices, dtype=torch.long)
                return full.index_select(0, idx)

    def _load_local_fused_experts(self, layer_idx):
        for canonical_name in self._expert_tensor_names(layer_idx):
            shard_path, ckpt_name = self._resolve_ckpt_tensor(canonical_name)

            local_tensor_cpu = self._read_first_dim_indices(
                shard_path=shard_path,
                tensor_name=ckpt_name,
                indices=self.local_expert_ids
            )

            local_tensor = local_tensor_cpu.to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True
            )

            self._set_tensor_allow_shape_change(canonical_name, local_tensor)

    def move_layer_to_gpu(self, layer_name):
        layer_idx = int(layer_name.split(".")[-1])
        prefixes = self._layer_prefixes(layer_name)
        non_mlp_pairs = self._names_from_ckpt(prefixes["non_mlp"])
        self._set_tensors(non_mlp_pairs)
        self._load_local_fused_experts(layer_idx)

    def offload_to_meta(self, layer_name):
        layer_idx = int(layer_name.split(".")[-1])
        prefixes = self._layer_prefixes(layer_name)
        non_mlp_pairs = self._names_from_ckpt(prefixes["non_mlp"])
        self._offload_names_to_meta(non_mlp_pairs)
        for name in self._expert_tensor_names(layer_idx):
            self._set_tensor_to_meta_allow_shape_change(name)

        torch.cuda.empty_cache()

    @torch.no_grad()
    def get_layer_activations(self, data_all):
        current_layer = self.get_layer_module(self.current_layer_idx)
        num_samples = data_all['input'].shape[0]
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, num_samples)
            x = data_all['input'][start_idx:end_idx].to(self.device, non_blocking=True)
            additional_layer_inputs = {"attention_mask": None}
            for k, v in self.kwargs.items():
                additional_layer_inputs[k] = v

            hidden_states = current_layer.input_layernorm(x)
            attn_out = self._run_attention(current_layer, self.current_layer_idx, hidden_states, additional_layer_inputs)
            mlp_input = x + attn_out
            out = self.run_expert_parallel(mlp_input)

            data_all['input'][start_idx:end_idx] = out.detach().cpu()

    @torch.no_grad()
    def get_mlp_output(self, mlp_input_batch):
        return self.run_expert_parallel(mlp_input_batch)

    def _dispatch_tokens(self, mlp_input_batch):
        layer = self.get_layer_module(self.current_layer_idx)
        device = self.device
        pg = dist.group.WORLD

        B, T, H = mlp_input_batch.shape

        hidden = layer.post_attention_layernorm(mlp_input_batch)
        x_flat = hidden.reshape(B * T, H)

        _, topw, topi = layer.mlp.gate(hidden)

        top_k = getattr(
            layer.mlp,
            "top_k",
            getattr(self.model.config, "num_experts_per_tok", 2)
        )

        tok_idx_flat = torch.arange(
            B * T,
            device=device,
            dtype=torch.long
        ).repeat_interleave(top_k)

        eid_flat = topi.reshape(-1).to(torch.long)
        w_flat = topw.reshape(-1).to(self.dtype)

        owners_flat = self._owner_lut[eid_flat]

        perm = torch.argsort(owners_flat, stable=True)

        owners_flat = owners_flat.index_select(0, perm)
        send_idx_flat = tok_idx_flat.index_select(0, perm)
        send_eid_flat = eid_flat.index_select(0, perm)
        send_w_flat = w_flat.index_select(0, perm)
        send_x_flat = x_flat.index_select(0, send_idx_flat)

        in_sizes_tensor = torch.bincount(
            owners_flat,
            minlength=self.world_size
        ).to(torch.long)

        all_sizes = [
            torch.empty_like(in_sizes_tensor)
            for _ in range(self.world_size)
        ]

        dist.all_gather(all_sizes, in_sizes_tensor, group=pg)

        recv_sizes = torch.stack(all_sizes)[:, self.rank]

        out_split_sizes = recv_sizes.tolist()
        in_split_sizes = in_sizes_tensor.tolist()

        xin = AllToAllTokens.apply(send_x_flat, out_split_sizes, in_split_sizes, pg)
        win = AllToAllTokens.apply(send_w_flat.unsqueeze(1), out_split_sizes, in_split_sizes, pg).to(self.dtype)
        eids = AllToAllTokens.apply(send_eid_flat.unsqueeze(1), out_split_sizes, in_split_sizes, pg).squeeze(1)

        return x_flat, hidden, send_idx_flat, in_split_sizes, out_split_sizes, xin, win, eids, B, T, H

    def _batched_expert_forward(self, xin, eids, quantized_weights=None):
        if xin.numel() == 0:
            return xin

        layer = self.get_layer_module(self.current_layer_idx)
        layer_key = self.get_current_layer()

        experts = layer.mlp.experts
        gate_up_proj = experts.gate_up_proj
        down_proj = experts.down_proj

        eids_long = eids.to(torch.long)
        unique_eids, inverse, counts = torch.unique(
            eids_long,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )

        sort_idx = torch.argsort(inverse, stable=True)
        sorted_x = xin.index_select(0, sort_idx)

        out_buf = torch.empty_like(sorted_x)

        offset = 0
        for i, eid_val in enumerate(unique_eids.tolist()):
            n = counts[i].item()
            inp_e = sorted_x[offset:offset + n]

            local_pos = int(self._global_to_local_lut[eid_val].item())
            if local_pos < 0:
                raise RuntimeError(
                    f"Rank {self.rank} received expert {eid_val}, "
                    f"but it does not own that expert."
                )

            base = f"{layer_key}.mlp.experts.{eid_val}"

            if quantized_weights is None:
                gate_up_w = gate_up_proj[local_pos]
                intermediate = gate_up_w.shape[0] // 2

                gate_w = gate_up_w[:intermediate]
                up_w = gate_up_w[intermediate:]
                down_w = down_proj[local_pos]
            else:
                gate_w = quantized_weights[base]["gate_proj"]
                up_w = quantized_weights[base]["up_proj"]
                down_w = quantized_weights[base]["down_proj"]

            gate_out = F.linear(inp_e, gate_w)
            up_out = F.linear(inp_e, up_w)
            hidden = F.silu(gate_out) * up_out
            out_e = F.linear(hidden, down_w)

            out_buf[offset:offset + n] = out_e
            offset += n

        unsort_idx = torch.argsort(sort_idx)
        return out_buf.index_select(0, unsort_idx)

    def _shared_expert_forward(self, hidden):
        layer = self.get_layer_module(self.current_layer_idx)

        shared = layer.mlp.shared_expert(hidden)
        if isinstance(shared, tuple):
            shared = shared[0]

        shared_gate = torch.sigmoid(layer.mlp.shared_expert_gate(hidden))
        return shared_gate * shared

    def run_expert_parallel(self, mlp_input_batch, quantized_weights=None):
        pg = dist.group.WORLD

        x_flat, hidden, send_idx_flat, in_split_sizes, out_split_sizes, xin, win, eids, B, T, H = \
            self._dispatch_tokens(mlp_input_batch)

        out_local = self._batched_expert_forward(xin, eids, quantized_weights)

        out_local = out_local * win

        returned = AllToAllTokens.apply(out_local, in_split_sizes, out_split_sizes, pg)

        y_flat = x_flat.new_zeros(x_flat.shape)
        y_flat.index_add_(0, send_idx_flat, returned)

        sparse_y = y_flat.view(B, T, H)
        shared_y = self._shared_expert_forward(hidden)

        return mlp_input_batch + sparse_y + shared_y

    def calculate_mse(self, mlp_input_batch, quantized_weights, self_attn=False, validation=False, accumulation_steps=1):
        with torch.no_grad():
            _, _, _, _, _, xin, _, eids, _, _, _ = self._dispatch_tokens(mlp_input_batch)

            out_fp = self._batched_expert_forward(xin, eids, quantized_weights=None)

        out_q = self._batched_expert_forward(xin, eids, quantized_weights)

        mse = self.loss_fn(out_q, out_fp)

        if not validation:
            (mse / accumulation_steps).backward()

        return mse.item()

    def _make_virtual_expert_linears(self, layer):
        experts = layer.mlp.experts

        gate_up_proj = experts.gate_up_proj
        down_proj = experts.down_proj

        intermediate = gate_up_proj.shape[1] // 2

        subset = {}

        for local_pos, global_eid in enumerate(self.local_expert_ids):
            base = f"{self.get_current_layer()}.mlp.experts.{global_eid}"

            subset[f"{base}.gate_proj"] = _VirtualLinear(
                gate_up_proj[local_pos, :intermediate]
            )
            subset[f"{base}.up_proj"] = _VirtualLinear(
                gate_up_proj[local_pos, intermediate:]
            )
            subset[f"{base}.down_proj"] = _VirtualLinear(
                down_proj[local_pos]
            )

        return subset

    def _add_gptq_batches_from_dispatched(self, xin, eids, gpts, virtual_linears):
        if xin.numel() == 0:
            return

        eids_long = eids.to(torch.long)
        unique_eids, inverse, counts = torch.unique(
            eids_long,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )

        sort_idx = torch.argsort(inverse, stable=True)
        sorted_x = xin.index_select(0, sort_idx)

        offset = 0
        for i, eid_val in enumerate(unique_eids.tolist()):
            n = counts[i].item()
            inp_e = sorted_x[offset:offset + n]

            local_pos = int(self._global_to_local_lut[eid_val].item())
            if local_pos < 0:
                raise RuntimeError(
                    f"Rank {self.rank} received expert {eid_val}, "
                    f"but it does not own that expert."
                )

            base = f"{self.get_current_layer()}.mlp.experts.{eid_val}"

            gate_key = f"{base}.gate_proj"
            up_key = f"{base}.up_proj"
            down_key = f"{base}.down_proj"

            gate_w = virtual_linears[gate_key].weight
            up_w = virtual_linears[up_key].weight
            down_w = virtual_linears[down_key].weight

            gate_out = F.linear(inp_e, gate_w)
            up_out = F.linear(inp_e, up_w)

            if gate_key in gpts:
                gpts[gate_key].add_batch(inp_e.data, gate_out.data)

            down_in = F.silu(gate_out) * up_out
            down_out = F.linear(down_in, down_w)

            if down_key in gpts:
                gpts[down_key].add_batch(down_in.data, down_out.data)

            offset += n

    def get_layer_initialization(self, trainer, gpt_all, config, logging):
        if logging is not None:
            logging = logging.logger

        layer_idx = self.current_layer_idx
        layer = self.get_layer_module(layer_idx)

        self.groupsize = config.quantization.groupsize

        subset = self._make_virtual_expert_linears(layer)

        skip_gptq = getattr(config.gptq, "skip", False)

        if skip_gptq:
            for name, module in subset.items():
                Q, scales = rtn_quantize(module, config, self.device, self.dtype)
                trainer.setup_layer_training(name, Q, scales)

            dist.barrier()
            return

        gpts = {}

        for name, module in subset.items():
            gpts[name] = GPTQ(module, name, config, self.device, self.dtype)

            if config.gptq.wbits < 16:
                gpts[name].quantizer = Quantizer()

                if getattr(config.quantization, "gsq_bits", None) == "nvfp4":
                    gpts[name].quantizer.configure(
                        4,
                        perchannel=True,
                        sym=True,
                        mse=True,
                        trits=False,
                        format="nvfp4",
                    )
                else:
                    gpts[name].quantizer.configure(
                        config.gptq.wbits,
                        perchannel=True,
                        sym=config.gptq.sym,
                        mse=True,
                        trits=config.gptq.trits,
                    )

        n_hessian = config.gptq.nsamples // self.world_size
        calib_report_interval = max(1, n_hessian // self.calib_report_divisor)
        calib_start = time.time()

        with torch.no_grad():
            for j in range(n_hessian):
                x = gpt_all["input"][j].unsqueeze(0).to(
                    self.device,
                    non_blocking=True,
                )
                x = self.get_mlp_input(x)
                _, _, _, _, _, xin, _, eids, _, _, _ = self._dispatch_tokens(x)

                self._add_gptq_batches_from_dispatched(
                    xin=xin,
                    eids=eids,
                    gpts=gpts,
                    virtual_linears=subset,
                )

                if self.rank == 0 and (j + 1) % calib_report_interval == 0:
                    report_gptq_calib(
                        j + 1,
                        n_hessian,
                        time.time() - calib_start,
                    )

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

        gptq_losses = []

        for name in list(gpts.keys()):
            if name.endswith(".up_proj"):
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

            if hasattr(gpts[name], "last_gptq_loss"):
                gptq_losses.append(gpts[name].last_gptq_loss)

            if scales is not None:
                trainer.setup_layer_training(name, Q, scales)
            else:
                self.update_quantized_weights(name, Q)

            if name.endswith(".gate_proj"):
                base = name[: -len(".gate_proj")]
                up_name = f"{base}.up_proj"

                gpts[up_name].H = gpts[name].H
                gpts[up_name].dead = gpts[name].dead

                Q, scales = gpts[up_name].fasterquant(
                    logging,
                    percdamp=config.gptq.percdamp,
                    blocksize=config.gptq.blocksize,
                    groupsize=config.gptq.groupsize,
                    static_groups=config.gptq.static_groups,
                    calculate_cholesky=False,
                    prunen=config.gptq.prunen,
                    prunem=config.gptq.prunem
                )

                if hasattr(gpts[up_name], "last_gptq_loss"):
                    gptq_losses.append(gpts[up_name].last_gptq_loss)

                if scales is not None:
                    trainer.setup_layer_training(up_name, Q, scales)
                else:
                    self.update_quantized_weights(up_name, Q)

                gpts[up_name].free()

            gpts[name].free()

        if gptq_losses:
            trainer.gptq_avg_loss = sum(gptq_losses) / len(gptq_losses)

        dist.barrier()

    def _safe_path_for_prefix(self, pfx):
        safe = pfx.replace(".", "_")
        return os.path.join(self.save_dir, f"{safe}.safetensors")

    def _load_plain_or_packed_tensor_file(self, path):
        tensors = safe_load_file(path, device="cpu")

        self.quantization_config.config_groups.group_0.weights.group_size = self.groupsize

        for name in tensors.keys():
            if (
                name.endswith(".weight_shape")
                or name.endswith(".weight_scale")
                or name.endswith("inv_freq")
            ):
                continue

            if name.endswith(".weight_packed"):
                base = name[: -len(".weight_packed")]

                compressed_data = {
                    "weight_packed": tensors[f"{base}.weight_packed"],
                    "weight_scale": tensors[f"{base}.weight_scale"],
                    "weight_shape": tensors[f"{base}.weight_shape"]
                }

                W_deq = self.compressor.decompress_weight(
                    compressed_data,
                    self.quantization_config.config_groups.group_0.weights
                )

                set_module_tensor_to_device(
                    self.model,
                    f"{base}.weight".replace(".language_model", ""),
                    self.device,
                    value=W_deq,
                    dtype=self.dtype
                )
            else:
                set_module_tensor_to_device(
                    self.model,
                    name.replace(".language_model", ""),
                    self.device,
                    value=tensors[name],
                    dtype=self.dtype
                )

    def _find_dequantized_virtual_weight(self, tensor_files, key_base):
        self.quantization_config.config_groups.group_0.weights.group_size = self.groupsize

        for tensors in tensor_files:
            packed_key = f"{key_base}.weight_packed"
            plain_key = f"{key_base}.weight"

            if packed_key in tensors:
                compressed_data = {
                    "weight_packed": tensors[f"{key_base}.weight_packed"],
                    "weight_scale": tensors[f"{key_base}.weight_scale"],
                    "weight_shape": tensors[f"{key_base}.weight_shape"]
                }

                return self.compressor.decompress_weight(
                    compressed_data,
                    self.quantization_config.config_groups.group_0.weights,
                )

            if plain_key in tensors:
                return tensors[plain_key]

        return None

    def _load_local_quantized_experts_from_disc(self, layer_idx):
        base = f"{self.layer_prefix}.{layer_idx}"

        candidate_prefixes = [
            base,
            f"{base}.mlp.experts"
        ]
        candidate_prefixes += [
            f"{base}.mlp.experts.{e}"
            for e in self.local_expert_ids
        ]

        tensor_files = []
        for pfx in candidate_prefixes:
            path = self._safe_path_for_prefix(pfx)
            if os.path.exists(path):
                tensor_files.append(safe_load_file(path, device="cpu"))

        gate_rows = []
        up_rows = []
        down_rows = []

        for e in self.local_expert_ids:
            expert_base = f"{base}.mlp.experts.{e}"

            gate_w = self._find_dequantized_virtual_weight(
                tensor_files,
                f"{expert_base}.gate_proj",
            )
            up_w = self._find_dequantized_virtual_weight(
                tensor_files,
                f"{expert_base}.up_proj",
            )
            down_w = self._find_dequantized_virtual_weight(
                tensor_files,
                f"{expert_base}.down_proj",
            )

            gate_rows.append(gate_w)
            up_rows.append(up_w)
            down_rows.append(down_w)

        gate_up = torch.stack(
            [
                torch.cat([g, u], dim=0)
                for g, u in zip(gate_rows, up_rows)
            ],
            dim=0
        ).to(device=self.device, dtype=self.dtype)

        down = torch.stack(down_rows, dim=0).to(device=self.device, dtype=self.dtype)

        self._set_tensor_allow_shape_change(
            f"{base}.mlp.experts.gate_up_proj",
            gate_up,
        )
        self._set_tensor_allow_shape_change(
            f"{base}.mlp.experts.down_proj",
            down,
        )

    def load_from_disc(self, layer_name):
        layer_idx = int(layer_name.split(".")[-1])
        prefixes = self._layer_prefixes(layer_name)

        for pfx in prefixes["non_mlp"]:
            path = self._safe_path_for_prefix(pfx)

            if os.path.exists(path):
                self._load_plain_or_packed_tensor_file(path)
            else:
                pairs = self._names_from_ckpt(pfx)
                self._set_tensors(pairs)

        self._load_local_quantized_experts_from_disc(layer_idx)

    def _load_layer_for_eval(self, layer_idx, read_from_disk):
        layer_name = f"{self.layer_prefix}.{layer_idx}"
        if layer_idx <= read_from_disk:
            self.load_from_disc(layer_name)
        else:
            self.move_layer_to_gpu(layer_name)

    @torch.no_grad()
    def ppl_evaluation(self, read_from_disk=-1):
        dataset = get_dataset("wikitext2", self.tokenizer)

        testloader = prepare_test_dataloader(
            dataset=dataset["test"],
            tokenizer=self.tokenizer,
            seqlen=self.model.seqlen,
            batch_size=1,
            world_size=self.world_size,
            rank=self.rank
        )

        pad_token_id = getattr(self.model.config, "pad_token_id", None)

        if pad_token_id is not None:
            loss_fn = torch.nn.CrossEntropyLoss(
                reduction="none",
                ignore_index=pad_token_id
            )
        else:
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        self.model.eval()

        self.move_embed_to(self.device)
        self.move_output_heads_to(self.device)

        input_ids_cpu_list = []
        activations_cpu_list = []

        for batch in testloader:
            ids_cpu = batch["input_ids"].to(
                "cpu",
                non_blocking=True
            ).pin_memory()

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
                report_ppl_layer(
                    i,
                    self.num_layers,
                    elapsed=time.time() - ppl_start
                )

            self.current_layer_idx = i
            layer_name = f"{self.layer_prefix}.{i}"

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

                activations_cpu_list[b] = x.to(
                    "cpu",
                    non_blocking=True
                ).pin_memory()

                del x

            self.offload_to_meta(layer_name)
            torch.cuda.synchronize(self.device)

            if i + 1 < self.num_layers:
                with torch.cuda.stream(transfer_stream):
                    self._load_layer_for_eval(i + 1, read_from_disk)
                transfer_stream.synchronize()

            torch.cuda.empty_cache()

        local_nll_sum = torch.tensor(0.0, device=self.device)
        local_tok_cnt = torch.tensor(0.0, device=self.device)

        self.move_embed_to("meta")

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

        dist.all_reduce(local_nll_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_tok_cnt, op=dist.ReduceOp.SUM)

        mean_nll = (local_nll_sum / local_tok_cnt).item()
        ppl = math.exp(mean_nll)

        torch.cuda.synchronize(self.device)
        gc.collect()
        torch.cuda.empty_cache()

        return ppl
