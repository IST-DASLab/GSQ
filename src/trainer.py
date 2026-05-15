import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import math
import time
from lion_pytorch import Lion
import torch.nn.functional as F
import torch.distributed as dist
from src.quantization import *
from src.utils.progress_reporter import report_gumbel_epoch, report_gumbel_step

class CUDAPrefetcher:
    def __init__(self, iterable, device):
        self.iterable = iter(iterable)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_batch = None
        self.preload()

    def preload(self):
        try:
            batch = next(self.iterable)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = batch.to(device=self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream(self.device).wait_stream(self.stream)

        batch = self.next_batch
        if batch is None:
            return None

        self.preload()
        return batch

class QuantizationTrainer:
    def __init__(self, model, config, dtype, self_attn=False):
        self.model = model
        self.config = config
        self.quantizers = {}
        self.optimizer = None
        self.scheduler = None
        self.device = model.device
        self.dtype = dtype
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.optimizer_params = []
        self.min_loss = float('inf')
        self.routing_cache = None
        self.batch_size = self.config.data.batch_size
        self.global_rank = getattr(self.model, 'rank', 0)
        self.world_size = getattr(self.model, 'world_size', 1)
        self.use_dist = self.world_size > 1
        if hasattr(self.model, 'save_dir'):
            self.model.save_dir = config.training.checkpoint_dir
        self.train_attn = self_attn

    def _pin_input_if_needed(self, data_dict):
        x = data_dict["input"]
        if x.device.type == "cpu" and not x.is_pinned():
            data_dict["input"] = x.pin_memory()

        return data_dict
        
    def _create_quantizer(self, Q, scales):
        gsq_bits = getattr(self.config.quantization, 'gsq_bits', 2)
        groupsize = self.config.quantization.groupsize
        std = self.config.quantization.std
        strength = self.config.quantization.strength

        logits_dtype_str = getattr(self.config.quantization, 'logits_dtype', None)
        if logits_dtype_str == "float32":
            logits_dtype = torch.float32
        else:
            logits_dtype = self.dtype

        if gsq_bits == 1:
            return GumbelQuantizer1Bit(Q, scales, groupsize, std, strength, self.device, self.dtype, logits_dtype)
        elif gsq_bits == 2:
            return GumbelQuantizer2Bit(Q, scales, groupsize, std, strength, self.device, self.dtype, logits_dtype)
        elif gsq_bits in [3, 4]:
            return GumbelQuantizerInt(Q, scales, groupsize, std, strength, self.device, self.dtype, logits_dtype, bits=gsq_bits)
        elif gsq_bits in ("ternary", "1.58", 1.58):
            return GumbelQuantizerTernary(Q, scales, groupsize, std, strength, self.device, self.dtype, logits_dtype)
        else:
            raise ValueError(
                f"Unsupported gsq_bits={gsq_bits!r}. "
                f"Supported: 1, 2, 3, 4, 'ternary' (aliases: '1.58')"
            )

    def setup_layer_training(self, tensor_name, Q, scales):
        quantizer = self._create_quantizer(Q, scales)

        if self.use_dist and not self.model.is_moe:
            for p in quantizer.parameters():
                dist.broadcast(p.data, src=0)

        self.quantizers[tensor_name] = quantizer

        logit_params = [p for n, p in quantizer.named_parameters() if n != 'scales']
        self.optimizer_params.extend([
            {'params': logit_params, 'lr': self.config.training.lr1,
            'weight_decay': self.config.training.weight_decay, 'lr_decay_tag': True},
            {'params': quantizer.scales, 'lr': self.config.training.lr2,
            'weight_decay': 0.0, 'lr_decay_tag': True}
        ])
        
    def train_layer(self, layer_name, train_all, val_all, logging,
                    layer_idx=None, num_layers=None):
        if not self.quantizers:
            logging.info(f"Skipping GSQ training for layer {layer_name} since no quantizers were set up.")
            return
        if logging is not None:
            logging = logging.logger

        train_all = self._pin_input_if_needed(train_all)
        val_all = self._pin_input_if_needed(val_all)

        num_epochs = self.config.training.num_epochs
        num_samples = train_all["input"].shape[0]
        val_num_samples = val_all["input"].shape[0]

        batch_size = self.config.data.batch_size // self.world_size
        batch_size = max(1, batch_size)

        self.optimizer = Lion(self.optimizer_params, betas=tuple(self.config.training.lion_betas))

        self.quant_params = []
        for _, quantizer in self.quantizers.items():
            self.quant_params.append(quantizer.quant_logits)

        steps_per_epoch = (num_samples + batch_size - 1) // batch_size
        num_training_steps = steps_per_epoch * num_epochs

        self.scheduler = CustomLRScheduler(
            self.optimizer,
            num_training_steps,
            self.config.training.warmup_steps,
            lr_decay_type=self.config.training.lr_decay_type,
            min_lr=self.config.training.scheduler_min_lr
        )

        initial_temperature, final_temperature = self.config.quantization.temperature
        initial_scale, final_scale = self.config.quantization.scale

        step = 0
        phase_start = time.time()
        step_report_interval = max(1, steps_per_epoch // self.config.logging.step_report_divisor)

        micro = max(1, self.config.training.device_microbatch_size)

        for epoch in range(num_epochs):
            epoch_start = time.time()

            if epoch == 0:
                temperature = initial_temperature
                scale = initial_scale

                val_soft_losses, val_hard_losses = self.run_validation_epoch(
                    val_all=val_all,
                    val_num_samples=val_num_samples,
                    batch_size=batch_size,
                    temperature=temperature,
                    scale=scale,
                    microbatch_size=micro
                )

                if self.global_rank == 0:
                    print(sum(val_hard_losses) / max(1, len(val_hard_losses)))

            epoch_losses = []

            batch_iter = self.make_cpu_batch_iter(
                x=train_all["input"],
                num_samples=num_samples,
                batch_size=batch_size,
                shuffle=True
            )

            prefetcher = CUDAPrefetcher(batch_iter, device=self.device)

            while True:
                batch_gpu = prefetcher.next()
                if batch_gpu is None:
                    break

                progress_denom = max(1, num_training_steps - 1)

                temperature = initial_temperature + (final_temperature - initial_temperature) * step / progress_denom

                scale = initial_scale + (final_scale - initial_scale) * step / progress_denom

                loss = self.train_step_gpu(
                    batch_gpu=batch_gpu,
                    temperature=temperature,
                    scale=scale,
                    microbatch_size=micro
                )

                epoch_losses.append(loss)

                if self.global_rank == 0:
                    report_gumbel_step(
                        step,
                        num_training_steps,
                        loss,
                        interval=step_report_interval
                    )

                if self.config.wandb.enabled and self.global_rank == 0:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    wandb.log(
                        {
                            "train/step_loss": loss,
                            "train/learning_rate": current_lr,
                            "train/temperature": temperature,
                            "train/scale": scale,
                            "train/global_step": step
                        }
                    )

                step += 1

            val_soft_losses, val_hard_losses = self.run_validation_epoch(
                val_all=val_all,
                val_num_samples=val_num_samples,
                batch_size=batch_size,
                temperature=temperature,
                scale=scale,
                microbatch_size=micro
            )

            avg_train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            avg_val_soft_loss = sum(val_soft_losses) / max(1, len(val_soft_losses))
            avg_val_hard_loss = sum(val_hard_losses) / max(1, len(val_hard_losses))

            epoch_time = time.time() - epoch_start
            phase_elapsed = time.time() - phase_start

            if self.global_rank == 0 and logging is not None:
                logging.info(
                    f"Layer {layer_name} - Epoch {epoch + 1}: "
                    f"Train Loss = {avg_train_loss:.2e}, "
                    f"Val Soft Loss = {avg_val_soft_loss:.2e}, "
                    f"Val Hard Loss = {avg_val_hard_loss:.2e}"
                )

                report_gumbel_epoch(
                    layer_name,
                    epoch,
                    num_epochs,
                    phase_elapsed,
                    avg_train_loss=avg_train_loss,
                    avg_val_loss=avg_val_hard_loss,
                    epoch_time=epoch_time,
                    temperature=temperature,
                    scale=scale
                )

            if self.config.wandb.enabled and self.global_rank == 0:
                wandb.log(
                    {
                        f"{layer_name}/train_loss": avg_train_loss,
                        f"{layer_name}/val_soft_loss": avg_val_soft_loss,
                        f"{layer_name}/val_hard_loss": avg_val_hard_loss,
                        f"{layer_name}/temperature": temperature,
                        f"{layer_name}/scale": scale,
                        f"{layer_name}/epoch": epoch + 1,
                        f"{layer_name}/epoch_time_sec": epoch_time
                    }
                )

        if self.use_dist:
            dist.barrier()

        for tensor_name, quantizer in self.quantizers.items():
            if self.train_attn or self.model.current_layer_idx == -1:
                self.model.update_quantized_weights(tensor_name, quantizer.get_hard_weights())
            else:
                if "gate_proj" in tensor_name:
                    base = tensor_name[: -len(".gate_proj")]
                    pairs = {
                        "gate_proj": quantizer.get_hard_weights(), 
                        "up_proj": self.quantizers[f"{base}.up_proj"].get_hard_weights(),
                        "down_proj": self.quantizers[f"{base}.down_proj"].get_hard_weights()
                    }
                    if self.model.is_moe or self.global_rank == 0:
                        self.model.save_to_disc(base, pairs)

        if self.use_dist:
            dist.barrier()
        self.quantizers.clear()

        del self.optimizer, self.optimizer_params, self.quantizers
        torch.cuda.empty_cache()
        
    def make_cpu_batch_iter(self, x, num_samples, batch_size, shuffle=True):
        if shuffle:
            perm = torch.randperm(num_samples)
        else:
            perm = torch.arange(num_samples)

        for i in range(0, num_samples, batch_size):
            yield x[perm[i:i + batch_size]]

    def run_validation_epoch(
        self,
        val_all,
        val_num_samples,
        batch_size,
        temperature,
        scale,
        microbatch_size
    ):
        val_soft_losses = []
        val_hard_losses = []

        batch_iter = self.make_cpu_batch_iter(
            x=val_all["input"],
            num_samples=val_num_samples,
            batch_size=batch_size,
            shuffle=False
        )

        prefetcher = CUDAPrefetcher(batch_iter, device=self.device)

        with torch.no_grad():
            while True:
                batch_gpu = prefetcher.next()
                if batch_gpu is None:
                    break

                val_soft_loss, val_hard_loss = self.validation_step_gpu(
                    batch_gpu=batch_gpu,
                    temperature=temperature,
                    scale=scale,
                    microbatch_size=microbatch_size
                )

                val_soft_losses.append(val_soft_loss)
                val_hard_losses.append(val_hard_loss)

        return val_soft_losses, val_hard_losses

    def train_step_gpu(self, batch_gpu, temperature, scale, microbatch_size):
        self.optimizer.zero_grad(set_to_none=True)

        batch_size = batch_gpu.shape[0]
        microbatch_size = max(1, microbatch_size)
        accumulation_steps = (batch_size + microbatch_size - 1) // microbatch_size

        total_loss = None

        for start in range(0, batch_size, microbatch_size):
            micro_batch = batch_gpu[start:start + microbatch_size]

            quantized_weights = self._build_weights(mode="soft", temperature=temperature, scale=scale)

            soft_loss = self.model.calculate_mse(
                micro_batch,
                quantized_weights,
                self.train_attn,
                accumulation_steps=accumulation_steps
            )

            loss = soft_loss / accumulation_steps

            if total_loss is None:
                total_loss = loss
            else:
                total_loss = total_loss + loss

        self.scheduler.step()

        if not self.model.is_moe and self.use_dist:
            self.average_grads()

        self.optimizer.step()

        quantized_weights.clear()

        if self.use_dist:
            pg = dist.group.WORLD
            t = torch.tensor(total_loss, device=self.device, dtype=torch.float32)

            if self.model.is_moe:
                dist.all_reduce(t, op=dist.ReduceOp.SUM, group=pg)
            else:
                dist.all_reduce(t, op=dist.ReduceOp.AVG, group=pg)

            return t.item()

        return total_loss.item()

    def _build_weights(self, mode, temperature, scale):
        weights = {}

        for tensor_name, quantizer in self.quantizers.items():
            if mode == "soft":
                w = quantizer.forward(temperature, scale)
            elif mode == "hard":
                w = quantizer.get_hard_weights()[0]
            else:
                raise ValueError(f"Unknown weight mode: {mode}")

            if self.use_dist and self.model.is_moe:
                prefix, leaf = tensor_name.rsplit(".", 1)
                if prefix not in weights:
                    weights[prefix] = {}
                weights[prefix][leaf] = w
            else:
                weights[tensor_name] = w

        return weights

    def validation_step_gpu(self, batch_gpu, temperature, scale, microbatch_size):
        batch_size = batch_gpu.shape[0]
        microbatch_size = max(1, microbatch_size)
        accumulation_steps = (batch_size + microbatch_size - 1) // microbatch_size

        total_soft_loss = None
        total_hard_loss = None

        soft_weights = self._build_weights(mode="soft", temperature=temperature, scale=scale)

        for start in range(0, batch_size, microbatch_size):
            micro_batch = batch_gpu[start:start + microbatch_size]

            loss = self.model.calculate_mse(
                micro_batch,
                soft_weights,
                self.train_attn,
                validation=True
            )

            loss = loss / accumulation_steps

            if total_soft_loss is None:
                total_soft_loss = loss
            else:
                total_soft_loss = total_soft_loss + loss

        soft_weights.clear()

        hard_weights = self._build_weights(mode="hard", temperature=temperature, scale=scale)

        for start in range(0, batch_size, microbatch_size):
            micro_batch = batch_gpu[start:start + microbatch_size]

            loss = self.model.calculate_mse(
                micro_batch,
                hard_weights,
                self.train_attn,
                validation=True
            )

            loss = loss / accumulation_steps

            if total_hard_loss is None:
                total_hard_loss = loss
            else:
                total_hard_loss = total_hard_loss + loss

        hard_weights.clear()

        if self.use_dist:
            pg = dist.group.WORLD

            t_soft = torch.tensor(total_soft_loss, device=self.device, dtype=torch.float32)
            t_hard = torch.tensor(total_hard_loss, device=self.device, dtype=torch.float32)

            if self.model.is_moe:
                dist.all_reduce(t_soft, op=dist.ReduceOp.SUM, group=pg)
                dist.all_reduce(t_hard, op=dist.ReduceOp.SUM, group=pg)
            else:
                dist.all_reduce(t_soft, op=dist.ReduceOp.AVG, group=pg)
                dist.all_reduce(t_hard, op=dist.ReduceOp.AVG, group=pg)

            return t_soft.item(), t_hard.item()

        return total_soft_loss.item(), total_hard_loss.item()

    def average_grads(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(self.world_size)

class CustomLRScheduler:
    def __init__(self, optimizer, total_steps, warmup_steps, lr_decay_type='linear', min_lr=0.0):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.lr_decay_type = lr_decay_type
        self.min_lr = min_lr
        self.current_step = 0

        self.initial_lrs = [group['lr'] for group in self.optimizer.param_groups]

    def step(self):
        for i, group in enumerate(self.optimizer.param_groups):
            tag = group.get('lr_decay_tag')
            init_lr = self.initial_lrs[i]

            if tag:
                group['lr'] = self._compute_lr(init_lr)
        
        self.current_step += 1

    def _compute_lr(self, base_lr):
        step = self.current_step
        if step < self.warmup_steps:
            return base_lr * (self.min_lr + (1 - self.min_lr) * step / self.warmup_steps)

        decay_step = step - self.warmup_steps
        decay_total = self.total_steps - 1 - self.warmup_steps
        if decay_total == 0:
            progress = 1
        else:
            progress = decay_step / decay_total

        if self.lr_decay_type == 'linear':
            return base_lr * (self.min_lr + (1 - self.min_lr) * (1 - progress))
        elif self.lr_decay_type == 'cosine':
            return base_lr * (self.min_lr + 0.5 * (1 - self.min_lr) * (1 + math.cos(math.pi * progress)))
        elif self.lr_decay_type == 'constant':
            return base_lr
        else:
            raise ValueError(f"Unknown lr_decay_type: {self.lr_decay_type}")
