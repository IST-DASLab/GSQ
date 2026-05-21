import logging
import math
import time
import torch
import torch.nn as nn
import transformers
from .quant import *
from ..quantization import *
import torch.distributed as dist
from lion_pytorch import Lion


DEBUG = False 


def rtn_quantize(layer, config, device, dtype):
    W = layer.weight.data.clone().float()
    rows, columns = W.shape
    groupsize = config.gptq.groupsize
    wbits = config.gptq.wbits
    sym = config.gptq.sym
    trits = getattr(config.gptq, 'trits', False)

    n_groups = (columns + groupsize - 1) // groupsize if groupsize > 0 else 1
    group_scales = torch.zeros(rows, n_groups, device=W.device, dtype=W.dtype)
    Q = torch.zeros_like(W)

    quantizer = Quantizer()
    quantizer.configure(wbits, perchannel=True, sym=sym, mse=True, trits=trits)

    for g in range(n_groups):
        c_start = g * groupsize
        c_end = min(c_start + groupsize, columns)
        W_group = W[:, c_start:c_end]

        quantizer.find_params(W_group, weight=True)
        group_scales[:, g] = quantizer.scale.view(rows)
        Q[:, c_start:c_end] = quantize(
            W_group, quantizer.scale, quantizer.zero, quantizer.maxq
        )

    return Q.to(device), group_scales.to(device)


class GPTQ:

    def __init__(self, layer, name, config, device, dtype):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.quantizer = None
        self.H = None
        self.dead = None
        self.nsamples = 0
        self.config = config
        self.device = device
        self.dtype = dtype

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if self.H is None:
            self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def cholesky(self, percdamp=.01):
        if self.H is None:
            logging.getLogger(__name__).warning(
                "GPTQ %s: no calibration batches (H is None); using identity Hessian fallback", self.name
            )
            self.H = torch.eye(self.columns, device=self.dev, dtype=torch.float32)
        self.dead = torch.diag(self.H) == 0
        self.H[self.dead, self.dead] = 1

        damp = percdamp * torch.mean(torch.diag(self.H))
        diag = torch.arange(self.columns, device=self.dev)
        self.H[diag, diag] += damp
        self.H = torch.linalg.cholesky(self.H)
        self.H = self.H.to(self.dtype)

    def sync_H(self, world_size):
        dist.all_reduce(self.H, op=dist.ReduceOp.SUM)
        self.H = self.H / world_size

    def fasterquant(
        self, logging, blocksize=128, percdamp=.01, groupsize=-1, static_groups=False, calculate_cholesky=True, prunen=0, prunem=0
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        if calculate_cholesky:
            self.cholesky(percdamp)
        W[:, self.dead] = 0

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        rows = W.shape[0]
        if groupsize == -1:
            n_groups = 1
        else:
            n_groups = (self.columns + groupsize - 1) // groupsize
        group_scales = torch.zeros(rows, n_groups, device=W.device, dtype=W.dtype)
        group_zeros  = torch.zeros(rows, n_groups, device=W.device, dtype=W.dtype)

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)
            for g, q in enumerate(groups):
                group_scales[:, g] = q.scale.view(rows)
                group_zeros[:, g]  = q.zero.view(rows)

        gsq_bits = self.config.quantization.gsq_bits
        if gsq_bits == "24int":
            possible_masks = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=self.dtype, device=self.device)
            possible_masks = torch.repeat_interleave(self.possible_masks, 2).reshape(-1, 8)
        else:
            possible_masks = None

        if gsq_bits not in ("ternary", "1.58", 1.58):
            total_loss = torch.zeros((), device=self.dev)
            Q = torch.zeros_like(W)

            H = torch.cholesky_inverse(self.H.float())
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H

            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                if prunen > 0:
                    mask1 = torch.zeros_like(W1, dtype=torch.bool)
                else:
                    mask1 = None

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if self.quantizer and groupsize != -1:
                        if not static_groups:
                            if (i1 + i) % groupsize == 0:
                                self.quantizer.find_params(
                                    W[:, (i1 + i):(i1 + i + groupsize)],
                                    weight=True
                                )
                                g = (i1 + i) // groupsize
                                group_scales[:, g] = self.quantizer.scale.view(rows)
                                group_zeros[:, g] = self.quantizer.zero.view(rows)
                        else:
                            idx = i1 + i
                            self.quantizer = groups[idx // groupsize]

                    if possible_masks is not None and (i % prunem == 0):
                        m = self.possible_masks.shape[1]
                        if i + m > count:
                            continue

                        block = W1[:, i:(i + m)]
                        d_block = torch.diag(Hinv1)[i:(i + m)]

                        scores = (block ** 2) / (d_block.reshape(1, -1) ** 2)
                        masks = self.possible_masks
                        loss_per_mask = torch.matmul(
                            scores, (1 - masks).T
                        )
                        best_mask_idx = torch.argmin(loss_per_mask, dim=1)
                        selected_masks = masks[best_mask_idx]
                        mask1[:, i:(i + m)] = (selected_masks == 0)
                    elif prunen > 0 and (i % prunem == 0):
                        m = min(prunem, count - i)
                        n = min(prunen, m)
                        block_scores = (W1[:, i:(i + m)] ** 2) / (
                            torch.diag(Hinv1)[i:(i + m)].reshape((1, -1)) ** 2
                        )
                        idx = torch.topk(block_scores, n, dim=1, largest=False).indices
                        mask1.scatter_(1, i + idx, True)

                    q = w.clone()
                    if mask1 is not None:
                        q[mask1[:, i]] = 0

                    if self.quantizer:
                        q = quantize(
                            q.unsqueeze(1),
                            self.quantizer.scale,
                            self.quantizer.zero,
                            self.quantizer.maxq,
                        ).flatten()

                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / (d ** 2)

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                W[:, i1:i2] = Q1
                total_loss += Losses1.sum() / 2
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            
            Q = W.clone()
            W = self.layer.weight.data.clone()
            gptq_loss = torch.linalg.norm((W - Q.to(self.dtype)) @ self.H, ord='fro')**2
            self.last_gptq_loss = gptq_loss.item()
            if logging is not None:
                logging.info(f'Layer {self.name}: GPTQ Loss = {self.last_gptq_loss}')

        if "q_proj" in self.name or "k_proj" in self.name or "in_proj_qkv" in self.name or gsq_bits in ("ternary", "1.58", 1.58):
            if gsq_bits == 1:
                quantizer = GumbelQuantizer1Bit(Q, group_scales, groupsize, self.config.quantization.std, self.config.quantization.strength, self.device, self.dtype)
            elif gsq_bits == 2:
                quantizer = GumbelQuantizer2Bit(Q, group_scales, groupsize, self.config.quantization.std, self.config.quantization.strength, self.device, self.dtype, logits_dtype=torch.float32)
            elif gsq_bits in ("ternary", "1.58", 1.58):
                quantizer = GumbelQuantizerTernary(torch.zeros_like(W), group_scales, groupsize, self.config.quantization.std, 0, self.device, self.dtype, logits_dtype=torch.float32)
            elif gsq_bits in [3, 4]:
                quantizer = GumbelQuantizerInt(Q, group_scales, groupsize, self.config.quantization.std, self.config.quantization.strength, self.device, self.dtype, logits_dtype=torch.float32)
            else:
                raise ValueError(
                    f"Unsupported gsq_bits={gsq_bits!r}. "
                    f"Supported: 1, 2, 3, 4, 'ternary' (aliases: '1.58')"
                )

            optimizer_params = []

            logit_params = [p for n, p in quantizer.named_parameters() if n != 'scales']
            optimizer_params.extend([
                {'params': logit_params, 'lr': self.config.training.lr1,
                'weight_decay': self.config.training.weight_decay},
                {'params': quantizer.scales, 'lr': self.config.training.lr2,
                'weight_decay': 0.0}
            ])

            optimizer = Lion(optimizer_params, betas=self.config.training.lion_betas)
            num_epochs = 2000

            initial_temperature, final_temperature = self.config.quantization.temperature
            initial_scale, final_scale = self.config.quantization.scale
            if gsq_bits in ("ternary", "1.58", 1.58):
                initial_temperature *= 2
            
            for epoch in range(num_epochs):
                temperature = initial_temperature + (final_temperature - initial_temperature) * (epoch / (num_epochs - 1))
                scale = initial_scale + (final_scale - initial_scale) * (epoch / (num_epochs - 1))
                W_q = quantizer.forward(temperature, scale)
                loss = torch.linalg.norm((W - W_q) @ self.H, ord='fro')**2
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            W_q, scales = quantizer.get_hard_weights()
            final_loss = torch.linalg.norm((W - W_q) @ self.H, ord='fro')**2
            if logging is not None:
                logging.info(f'Layer {self.name}: '
                                f'Train Loss = {final_loss.item():.3f}')

            optimizer_params.clear()
            
            return W_q, scales

        return Q, group_scales

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
