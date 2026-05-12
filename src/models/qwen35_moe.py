import time
import torch
import math
import gc
from .base import BaseModelWrapper
from src.evaluation.wiki_eval import *
from src.utils.progress_reporter import report_ppl_layer


class Qwen35MoeWrapper(BaseModelWrapper):
    def __init__(self, model_name, tokenizer, batch_size, seqlen, device, dtype):
        super().__init__(model_name, tokenizer, batch_size, seqlen, device, dtype)
        self.layer_prefix = "model.language_model.layers"
        self.num_layers = len(self.model.model.layers)
        self.num_experts = self.model.config.num_experts
        self.layer_types = getattr(self.model.config, 'layer_types', ["full_attention"] * self.num_layers)
        self.is_moe = True

    def _is_linear_attention_layer(self, layer_idx):
        if layer_idx < len(self.layer_types):
            return self.layer_types[layer_idx] == "linear_attention"
        return False

    def _layer_prefixes(self, layer_name):
        layer_idx = int(layer_name.split('.')[-1])
        base = f"{self.layer_prefix}.{layer_idx}"

        attn_prefix = f"{base}.linear_attn" if self._is_linear_attention_layer(layer_idx) else f"{base}.self_attn"

        non_mlp = [
            f"{base}.input_layernorm",
            attn_prefix,
            f"{base}.mlp.gate",
            f"{base}.mlp.shared_expert",
            f"{base}.mlp.shared_expert_gate",
            f"{base}.post_attention_layernorm"
        ]
        mlp = [
            f"{base}.mlp.experts"
        ]
        return {"non_mlp": non_mlp, "mlp": mlp}

    def _run_attention(self, layer, layer_idx, hidden_states, additional_inputs):
        if self._is_linear_attention_layer(layer_idx):
            return layer.linear_attn(hidden_states)
        else:
            out = layer.self_attn(hidden_states, **additional_inputs)
            if isinstance(out, tuple):
                return out[0]
            return out

    def get_mlp_input(self, batch):
        current_layer = self.get_layer_module(self.current_layer_idx)

        additional_layer_inputs = {"attention_mask": None}
        for k, v in self.kwargs.items():
            additional_layer_inputs[k] = v

        hidden_states = current_layer.input_layernorm(batch)
        attn_out = self._run_attention(current_layer, self.current_layer_idx, hidden_states, additional_layer_inputs)
        return attn_out + batch

    def get_mlp_output(self, mlp_input_batch):
        current_layer = self.get_layer_module(self.current_layer_idx)

        hidden_states = current_layer.post_attention_layernorm(mlp_input_batch)
        hidden_states = current_layer.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        return hidden_states + mlp_input_batch

    def get_layer_module(self, idx):
        return self.model.model.layers[idx]

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
        else:
            self._offload_names_to_meta(names)

    @torch.no_grad()
    def ppl_evaluation(self, read_from_disk=-1):
        dataset = get_dataset("wikitext2", self.tokenizer)
        testloader = prepare_test_dataloader(
            dataset=dataset["test"],
            tokenizer=self.tokenizer,
            seqlen=self.model.seqlen,
            batch_size=1,
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

        mean_nll = (nll_sum / tok_cnt).item()
        ppl = math.exp(mean_nll)

        torch.cuda.synchronize(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        return ppl
