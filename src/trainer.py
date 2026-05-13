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
from src.utils.progress_reporter import report_gumbel_epoch, report_gumbel_step, report_pipeline

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
        if logging is not None:
            logging = logging.logger
        num_epochs = self.config.training.num_epochs
        num_samples = train_all['input'].shape[0]
        batch_size = self.config.data.batch_size // self.world_size

        self.optimizer = Lion(self.optimizer_params, betas=tuple(self.config.training.lion_betas))

        num_training_steps = (num_samples + batch_size - 1) // batch_size * num_epochs
        steps_per_epoch = (num_samples + batch_size - 1) // batch_size
        self.scheduler = CustomLRScheduler(
            self.optimizer, num_training_steps,
            self.config.training.warmup_steps,
            lr_decay_type=self.config.training.lr_decay_type,
            min_lr=self.config.training.scheduler_min_lr,
        )
        
        initial_temperature, final_temperature = self.config.quantization.temperature
        initial_scale, final_scale = self.config.quantization.scale

        step = 0
        phase_start = time.time()
        step_report_interval = max(1, steps_per_epoch // self.config.logging.step_report_divisor)

        micro = max(1, self.config.training.device_microbatch_size)
        for epoch in range(num_epochs):
            t = epoch / (num_epochs - 1) if num_epochs > 1 else 0.0
            temperature = initial_temperature + (final_temperature - initial_temperature) * t
            scale = initial_scale + (final_scale - initial_scale) * t

            if epoch == 0:
                val_soft_losses, val_hard_losses = [], []
                
                num_batches = (val_all['input'].shape[0] + batch_size - 1) // batch_size
                
                with torch.no_grad():
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx*batch_size
                        end_idx = min((batch_idx+1)*batch_size,num_samples)
                        
                        val_soft_loss, val_hard_loss = self.validation_step(
                            val_all['input'][start_idx:end_idx],
                            temperature, 
                            scale,
                            micro
                        )
                        val_soft_losses.append(val_soft_loss)
                        val_hard_losses.append(val_hard_loss)

            epoch_losses = []
            epoch_start = time.time()

            for indices in self.get_random_batch_indices(num_samples, batch_size):
                temperature = initial_temperature + (final_temperature - initial_temperature) * step / (num_training_steps - 1)
                scale = initial_scale + (final_scale - initial_scale) * step / (num_training_steps - 1)
                loss = self.train_step(
                    train_all['input'][indices],
                    temperature, 
                    scale,
                    micro
                )
                epoch_losses.append(loss)

                if self.global_rank == 0:
                    report_gumbel_step(step, num_training_steps, loss,
                                       interval=step_report_interval)

                if self.config.wandb.enabled and self.global_rank == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    wandb.log({
                        "train/step_loss": loss,
                        "train/learning_rate": current_lr,
                        "train/temperature": temperature,
                        "train/scale": scale,
                        "train/global_step": step,
                    })

                step += 1

            val_soft_losses, val_hard_losses = [], []
            
            num_batches = (val_all['input'].shape[0] + batch_size - 1) // batch_size
            
            with torch.no_grad():
                for batch_idx in range(num_batches):
                    start_idx = batch_idx*batch_size
                    end_idx = min((batch_idx+1)*batch_size,val_all['input'].shape[0])
                    
                    val_soft_loss, val_hard_loss = self.validation_step(
                        val_all['input'][start_idx:end_idx],
                        temperature, 
                        scale,
                        micro
                    )
                    val_soft_losses.append(val_soft_loss)
                    val_hard_losses.append(val_hard_loss)
            
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            avg_val_soft_loss = sum(val_soft_losses) / len(val_soft_losses)
            avg_val_hard_loss = sum(val_hard_losses) / len(val_hard_losses)
            epoch_time = time.time() - epoch_start
            phase_elapsed = time.time() - phase_start

            if self.global_rank == 0:
                logging.info(
                    f'Layer {layer_name} - Epoch {epoch+1}: '
                    f'Train Loss = {avg_train_loss:.2e}, '
                    f'Val Soft Loss = {avg_val_soft_loss:.2e}, '
                    f'Val Hard Loss = {avg_val_hard_loss:.2e}'
                )
                report_gumbel_epoch(
                    layer_name, epoch, num_epochs, phase_elapsed,
                    avg_train_loss=avg_train_loss,
                    avg_val_loss=avg_val_hard_loss,
                    epoch_time=epoch_time,
                    temperature=temperature,
                    scale=scale,
                )
            
            if self.config.wandb.enabled and self.global_rank == 0:
                wandb.log({
                    f"{layer_name}/train_loss": avg_train_loss,
                    f"{layer_name}/val_soft_loss": avg_val_soft_loss,
                    f"{layer_name}/val_hard_loss": avg_val_hard_loss,
                    f"{layer_name}/temperature": temperature,
                    f"{layer_name}/scale": scale,
                    f"{layer_name}/epoch": epoch + 1,
                    f"{layer_name}/epoch_time_sec": epoch_time,
                })

            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

        if self.use_dist:
            if logging is not None and self.global_rank == 0:
                report_pipeline(f"Syncing ranks before writing checkpoints ({layer_name})")
            dist.barrier()

        if logging is not None and self.global_rank == 0:
            if self.train_attn:
                report_pipeline(f"Applying quantized attention weights in-place ({layer_name})")
            else:
                report_pipeline(f"Writing quantized shard(s) to disk ({layer_name})")

        for tensor_name, quantizer in self.quantizers.items():
            if self.train_attn:
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

        if logging is not None and self.global_rank == 0:
            report_pipeline(f"Finished writing layer checkpoint ({layer_name})")

        if self.use_dist:
            if logging is not None and self.global_rank == 0:
                report_pipeline("Shard writes done; syncing ranks before next phase")
            dist.barrier()
        self.quantizers.clear()

        del self.optimizer, self.optimizer_params, self.quantizers
        torch.cuda.empty_cache()
        
    def train_step(self, batch, temperature, scale, microbatch_size):
        self.optimizer.zero_grad(set_to_none=True)

        batch_size, seq_len, hidden_dim = batch.shape
        accumulation_steps = max(1, batch_size // microbatch_size)

        total_loss = 0.0

        for i in range(accumulation_steps):
            micro_batch = batch[i*microbatch_size:(i+1)*microbatch_size]

            quantized_weights = {}
            for tensor_name, quantizer in self.quantizers.items():
                if self.use_dist and self.model.is_moe:
                    prefix, leaf = tensor_name.rsplit(".", 1)
                    if prefix not in quantized_weights:
                        quantized_weights[prefix] = {}
                    quantized_weights[prefix][leaf] = quantizer.forward(temperature, scale)
                else:
                    quantized_weights[tensor_name] = quantizer.forward(temperature, scale)

            soft_loss = self.model.calculate_mse(
                micro_batch.to(self.device), quantized_weights, self.train_attn,
                accumulation_steps=accumulation_steps
            )
            total_loss += soft_loss / accumulation_steps

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
            total_loss = t.item()
        
        return total_loss

    def _build_weights(self, mode, temperature, scale):
        weights = {}
        for tensor_name, quantizer in self.quantizers.items():
            if mode == 'soft':
                w = quantizer.forward(temperature, scale)
            else:
                w = quantizer.get_hard_weights()[0]
            if self.use_dist and self.model.is_moe:
                prefix, leaf = tensor_name.rsplit(".", 1)
                if prefix not in weights:
                    weights[prefix] = {}
                weights[prefix][leaf] = w
            else:
                weights[tensor_name] = w
        return weights

    def validation_step(self, batch, temperature, scale, microbatch_size):
        batch_size, seq_len, hidden_dim = batch.shape
        microbatch_size = max(1, microbatch_size)
        accumulation_steps = max(1, batch_size // microbatch_size)

        total_soft_loss = 0.0
        total_hard_loss = 0.0

        soft_weights = self._build_weights('soft', temperature, scale)
        for i in range(accumulation_steps):
            micro_batch = batch[i*microbatch_size:(i+1)*microbatch_size].to(self.device)
            total_soft_loss += self.model.calculate_mse(micro_batch, soft_weights, self.train_attn, validation=True) / accumulation_steps
        soft_weights.clear()

        hard_weights = self._build_weights('hard', temperature, scale)
        for i in range(accumulation_steps):
            micro_batch = batch[i*microbatch_size:(i+1)*microbatch_size].to(self.device)
            total_hard_loss += self.model.calculate_mse(micro_batch, hard_weights, self.train_attn, validation=True) / accumulation_steps
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
            total_soft_loss = t_soft.item()
            total_hard_loss = t_hard.item()

        return total_soft_loss, total_hard_loss
    
    def get_random_batch_indices(self, num_samples, batch_size):
        perm = torch.randperm(num_samples)
        if self.use_dist:
            perm = perm.to(self.device)
            dist.broadcast(perm, src=0)
            perm = perm.cpu()
        for i in range(0, num_samples, batch_size):
            yield perm[i:i+batch_size]

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
