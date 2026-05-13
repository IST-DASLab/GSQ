import time
import torch
import torch.nn as nn
import torch.distributed as dist
from abc import ABC, abstractmethod
import json
from pathlib import Path
from types import SimpleNamespace
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig, AutoModelForCausalLM  
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from compressed_tensors import PackedQuantizationCompressor
from contextlib import ExitStack
from src.prior.sparsegpt import *
from src.prior.gptq import *
from src.evaluation.wiki_eval import *
from src.utils.progress_reporter import report_gptq_calib, report_gptq_linear

class BaseModelWrapper(ABC):
    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype, dummy=False):
        self.ckpt_path = self.resolve_model_path(model_name)
        self.dtype = self.normalize_dtype(dtype)
        self.device = device
        self.save_dir = None

        cfg = AutoConfig.from_pretrained(self.ckpt_path, trust_remote_code=True)
        # HF only sets _attn_implementation on the top-level config; propagate to
        # sub-configs so decoder layers pick up DeepseekV3FlashAttention2.
        if hasattr(cfg, 'text_config') and cfg.text_config is not None:
            cfg.text_config._attn_implementation = "sdpa"
            with init_empty_weights():
                empty_model = AutoModelForCausalLM.from_config(cfg.text_config, trust_remote_code=True).eval()
        else:
            with init_empty_weights():
                empty_model = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa", trust_remote_code=True).eval()

        self.model = empty_model

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.model.seqlen = seqlen
        self.model.config.use_cache = False

        self.current_layer_idx = 0
        self.loss_fn = torch.nn.MSELoss()

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.dummy = dummy
        self.kwargs = {}
        self._attention_mask_1 = None  # single-item causal mask, expanded on demand
        self.meta_init_std = 0.02
        self.calib_report_divisor = 10
        self.batch_report_divisor = 5

        if dummy:
            self._name_to_shard = None
            self._single_shard = None
        else:
            self._name_to_shard, self._single_shard = self._load_safetensor_index(self.ckpt_path)
        self.compressor = PackedQuantizationCompressor()

        self.quantization_config = self.dict_to_ns({
            "config_groups": {
              "group_0": {
                "input_activations": None,
                "output_activations": None,
                "targets": [
                  "Linear"
                ],
                "weights": {
                  "actorder": None,
                  "block_structure": None,
                  "dynamic": False,
                  "group_size": 32,
                  "num_bits": 4,
                  "observer": "minmax",
                  "observer_kwargs": {},
                  "strategy": "group",
                  "symmetric": True,
                  "type": "int"
                }
              }
            },
            "format": "pack-quantized",
            "ignore": [
              "lm_head",
              "re:.*self_attn.*"
            ],
            "kv_cache_scheme": None,
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed"
          })
        self.temp_weights = {}
        self.is_moe = False

    def resolve_model_path(self, model_name):
        p = Path(model_name)

        if p.exists():
            if p.is_file():
                return str(p.parent.resolve())
            return str(p.resolve())

        try:
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(repo_id=model_name, ignore_patterns=["*.bin", "*.pth"])
            return str(Path(local_dir).resolve())
        except Exception as e:
            raise FileNotFoundError(
                f"Could not resolve '{model_name}' as a local path or download it from the Hugging Face Hub. "
                f"Original error: {e}"
            )
        
    def dict_to_ns(self, d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: self.dict_to_ns(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [self.dict_to_ns(x) for x in d]
        else:
            return d
        
    @staticmethod
    def _load_safetensor_index(ckpt_path):
        index_json = os.path.join(ckpt_path, "model.safetensors.index.json")
        if os.path.exists(index_json):
            with open(index_json, "r") as f:
                idx = json.load(f)
            return idx["weight_map"], None

        single = os.path.join(ckpt_path, "model.safetensors")
        if os.path.exists(single):
            return None, single

        raise FileNotFoundError(f"No safetensors found under {ckpt_path}")

    def _materialize_to_device(self, name_shard_pairs):
        params = dict(self.model.named_parameters())
        for n, _ in name_shard_pairs:
            if n not in params:
                continue
            if params[n].device.type == "meta":
                t = torch.randn(params[n].shape, dtype=self.dtype, device=self.device) * self.meta_init_std
                set_module_tensor_to_device(self.model, n, self.device, value=t, dtype=self.dtype)

    def _names_from_ckpt(self, prefixes):
        if isinstance(prefixes, str):
            prefixes = [prefixes]

        def wanted(name):
            for p in prefixes:
                if name == p or p in name:
                    return True
            return False

        if self.dummy:
            return [(n, None) for n, _ in self.model.named_parameters() if wanted(n)]

        pairs = []
        if self._name_to_shard is None:
            names = safe_load_file(self._single_shard, device="cpu").keys()
            for n in names:
                if wanted(n):
                    pairs.append((n, self._single_shard))
        else:
            for n, shard in self._name_to_shard.items():
                if wanted(n):
                    pairs.append((n, os.path.join(self.ckpt_path, shard)))
        return pairs

    def _set_tensors(self, name_shard_pairs):
        if self.dummy:
            self._materialize_to_device(name_shard_pairs)
            return

        by_shard = {}
        for n, s in name_shard_pairs:
            by_shard.setdefault(s, []).append(n)

        for shard, names in by_shard.items():
            with safe_open(shard, framework="pt", device=str(self.device)) as f:
                for ckpt_name in names:
                    if ckpt_name.endswith("inv_freq"):
                        continue
                    t = f.get_tensor(ckpt_name)
                    model_name = ckpt_name.replace(".language_model", "")
                    set_module_tensor_to_device(self.model, model_name, self.device, value=t, dtype=t.dtype)

    def move_layer_to_gpu(self, layer_name):
        prefixes = self._layer_prefixes(layer_name)
        pairs = self._names_from_ckpt(prefixes["non_mlp"] + prefixes["mlp"])
        self._set_tensors(pairs)

    def _offload_names_to_meta(self, name_shard_pairs):
        names = [n if isinstance(n, str) else n[0] for n in name_shard_pairs]
        for n in names:
            if n.endswith("inv_freq"):
                continue
            n = n.replace(".language_model", "")
            set_module_tensor_to_device(self.model, n, "meta")
        torch.cuda.empty_cache()

    def offload_to_meta(self, layer_name):
        prefixes = self._layer_prefixes(layer_name)
        pairs = self._names_from_ckpt(prefixes["non_mlp"] + prefixes["mlp"])
        self._offload_names_to_meta(pairs)

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
                        self._attention_mask_1 = v.detach().cpu()
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

        self.kwargs["position_embeddings"] = (self.kwargs["position_embeddings"][0][0].unsqueeze(0), self.kwargs["position_embeddings"][1][0].unsqueeze(0))

    def _build_layer_inputs(self, batch_size):
        """Build kwargs dict for a layer forward, expanding attention_mask to batch_size."""
        inputs = dict(self.kwargs)
        inputs["attention_mask"] = None
        return inputs

    @torch.no_grad()
    def get_layer_activations(self, data_all):
        current_layer = self.get_layer_module(self.current_layer_idx)
        for batch_idx in range((data_all['input'].size(0) + self.batch_size - 1) // self.batch_size):
            start_idx, end_idx = batch_idx * self.batch_size, min((batch_idx + 1) * self.batch_size, data_all['input'].shape[0])
            x = data_all['input'][start_idx:end_idx].to(self.device, non_blocking=True)

            additional_layer_inputs = self._build_layer_inputs(x.shape[0])
            out = current_layer(x, **additional_layer_inputs)
            if isinstance(out, tuple):
                out = out[0]
            data_all['input'][start_idx:end_idx] = out.detach().cpu()

    @torch.no_grad()
    def get_mlp_input_all(self, data_all):
        num_samples = data_all['input'].shape[0]
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        for batch_idx in range(num_batches):
            start_idx, end_idx = batch_idx * self.batch_size, min((batch_idx + 1) * self.batch_size, num_samples)
            data_all['input'][start_idx:end_idx] = self.get_mlp_input(data_all['input'][start_idx:end_idx].to(self.device)).detach().cpu()
    
    @torch.no_grad()
    def get_mlp_output_all(self, data_all):
        num_samples = data_all['input'].shape[0]
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        for batch_idx in range(num_batches):
            start_idx, end_idx = batch_idx * self.batch_size, min((batch_idx + 1) * self.batch_size, num_samples)
            data_all['input'][start_idx:end_idx] = self.get_mlp_output(data_all['input'][start_idx:end_idx].to(self.device)).detach().cpu()
        
    @torch.no_grad()
    def get_loss(self, layer_inputs, layer_outputs):
        new_outputs = self.get_mlp_output(layer_inputs)
        val_loss = self.loss_fn(layer_outputs, new_outputs)
        return layer_inputs.detach().cpu(), new_outputs.detach().cpu(), val_loss.item()

    def get_layer_initialization(self, trainer, gpt_all, config, logging, is_attn=False):
        if logging is not None:
            logging = logging.logger
        current_layer = self.get_layer_module(self.current_layer_idx)
        self.groupsize = config.quantization.groupsize
        subset = {}
        
        if is_attn:
            self_attn = current_layer.self_attn
            base_prefix = f"{self.get_current_layer()}.self_attn"
            for name, module in self_attn.named_modules():
                if isinstance(module, torch.nn.Linear):
                    subset[f"{base_prefix}.{name}"] = module
        else:
            mlp = current_layer.mlp
            base_prefix = f"{self.get_current_layer()}.mlp"
            for name, module in mlp.named_modules():
                if isinstance(module, torch.nn.Linear):
                    subset[f"{base_prefix}.{name}"] = module
        
        skip_gptq = getattr(config.gptq, 'skip', False)

        if skip_gptq:
            for name in subset:
                Q, scales = rtn_quantize(subset[name], config, self.device, self.dtype)
                if "q_proj" in name or "k_proj" in name:
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
            if "q_proj" in name or "k_proj" in name:
                self.update_quantized_weights(name, (Q, scales))
                if self.world_size > 1:
                    layer = self._get_layer_by_name(name.replace(".language_model", ""))
                    dist.broadcast(layer.weight.data, src=0)
            else:
                trainer.setup_layer_training(name, Q, scales)
            gpts[name].free()

        if gptq_losses:
            trainer.gptq_avg_loss = sum(gptq_losses) / len(gptq_losses)

    def calculate_mse(self, batch, quantized_weights, self_attn, validation=False, accumulation_steps=1):
        with torch.no_grad():
            out_fp = self.forward_with_quantized(batch, None, self_attn)
        out_q = self.forward_with_quantized(batch, quantized_weights, self_attn)

        mse = self.loss_fn(out_q, out_fp)
        if not validation:
            (mse / accumulation_steps).backward()

        return mse.item()

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
                    layer = current_layer.self_attn
                else:
                    layer = current_layer.mlp

                for name, module in layer.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        if self_attn:
                            key = f"{self.get_current_layer()}.self_attn.{name}"
                        else: 
                            key = f"{self.get_current_layer()}.mlp.{name}"
                        if key in quantized_weights:
                            hooks.append(LinearWeightHook(module, quantized_weights[key]))
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

    def find_layers(self, module, layers=[nn.Linear], name=''):
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(self.find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res
        
    def _get_layer_by_name(self, layer_name):
        return self.model.get_submodule(layer_name)
        
    def update_quantized_weights(self, layer_name, quantized_weights):
        layer = self._get_layer_by_name(layer_name.replace(".language_model", ""))
        self.temp_weights[layer_name] = layer.weight.data
        self.temp_weights[f"{layer_name}.scale"] = quantized_weights[1]
        with torch.no_grad():
            layer.weight.data = quantized_weights[0].to(self.device).to(layer.weight.data.dtype)
        
    def get_current_layer(self):
        return f"{self.layer_prefix}.{self.current_layer_idx}"
        
    def move_to_next_layer(self):
        self.current_layer_idx += 1
        if self.current_layer_idx < self.num_layers:
            return self.get_current_layer()
        return None
    
    def normalize_dtype(self, x):
        if isinstance(x, torch.dtype):
            return x
        if isinstance(x, str):
            try:
                return getattr(torch, x)
            except AttributeError:
                raise ValueError(f"Unknown dtype string: {x}")
        raise TypeError(f"Expected torch.dtype or str, got {type(x)}")
    
    def save_to_disc(self, pfx, pairs):
        os.makedirs(self.save_dir, exist_ok=True)
        to_save = {}
        for name, pair in pairs.items():
            self.quantization_config.config_groups.group_0.weights.group_size = self.groupsize
            if isinstance(pair[1], tuple):
                compressed = self.compressor.compress_weight(
                    weight=pair[0].detach().cpu(),
                    scale=pair[1][0].detach().cpu(),
                    quantization_args=self.quantization_config.config_groups.group_0.weights
                )
            else:
                compressed = self.compressor.compress_weight(
                    weight=pair[0].detach().cpu(),
                    scale=pair[1].detach().cpu(),
                    quantization_args=self.quantization_config.config_groups.group_0.weights
                )
            base = f"{pfx}.{name}"
            to_save[base + ".weight_packed"] = compressed["weight_packed"]
            if isinstance(pair[1], tuple):
                to_save[base + ".weight_scale"]  = pair[1][0].detach().cpu()
            else:
                to_save[base + ".weight_scale"]  = pair[1].detach().cpu()
            to_save[base + ".weight_shape"]  = compressed["weight_shape"]
        safe = pfx.replace('.', '_')
        path = os.path.join(self.save_dir, f"{safe}.safetensors")
        safe_save_file(to_save, path)

    def save_attention_to_disc(self, pfx):
        os.makedirs(self.save_dir, exist_ok=True)
        to_save = {}
        self_attn_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        self.quantization_config.config_groups.group_0.weights.group_size = self.groupsize
        for name in self_attn_layers:
            base = f"{pfx}.{name}"
            module = self.model.get_submodule(base)
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
            module = self.model.get_submodule(pfx.replace(".language_model", ""))
            sd = module.state_dict(keep_vars=True)
            to_save = {}
            for local_name, tensor in sd.items():
                if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                    to_save[f"{pfx}.{local_name}"] = tensor.detach().cpu()
            safe = pfx.replace('.', '_')
            path = os.path.join(self.save_dir, f"{safe}.safetensors")
            safe_save_file(to_save, path)

    def load_from_disc(self, layer_name):
        prefixes = self._layer_prefixes(layer_name)
        if isinstance(prefixes, str):
            prefixes = [prefixes]

        files = {}
        for item in prefixes:
            for p in prefixes[item]:
                base = p.replace('.', '_')
                st = os.path.join(self.save_dir, f"{base}.safetensors")
                files[p] = st

        for p, path in files.items():
            tensors = safe_load_file(path, device="cpu")
            for name in tensors.keys():
                if name.endswith(".weight_shape") or name.endswith(".weight_scale") or name.endswith("inv_freq"):
                    continue
                if name.endswith(".weight_packed"):
                    base = name[: -len(".weight_packed")]
                    compressed_data = {
                        "weight_packed": tensors[f"{base}.weight_packed"], 
                        "weight_scale": tensors[f"{base}.weight_scale"],
                        "weight_shape": tensors[f"{base}.weight_shape"]
                    }
                    W_deq = self.compressor.decompress_weight(compressed_data, self.quantization_config.config_groups.group_0.weights)

                    set_module_tensor_to_device(self.model, f"{base}.weight".replace(".language_model", ""), self.device, value=W_deq, dtype=self.dtype)
                    continue
                set_module_tensor_to_device(self.model, name.replace(".language_model", ""), self.device, value=tensors[name], dtype=self.dtype)
        current_layer = self.get_layer_module(self.current_layer_idx)

    @abstractmethod
    def get_mlp_input(self, layer_input):
        pass

    @abstractmethod
    def get_mlp_output(self, mlp_input_batch):
        pass
    
    @abstractmethod
    def get_layer_module(self, idx):
        pass

    @abstractmethod
    def move_embed_to(self, device):
        pass

    @abstractmethod
    def move_output_heads_to(self, device):
        pass

    @abstractmethod
    def _layer_prefixes(self, layer_name):
        pass

    @abstractmethod
    def ppl_evaluation(self, read_from_disk=-1):
        pass
