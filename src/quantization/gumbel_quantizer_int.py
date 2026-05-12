import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelQuantizerInt(nn.Module):
    def __init__(self, Q, scales, groupsize, std, strength, device, dtype, logits_dtype=None, bits=3):
        super().__init__()
        self.weight_shape = tuple(Q.shape)
        self.device = device
        self.dtype = dtype
        self.logits_dtype = logits_dtype if logits_dtype is not None else dtype
        self.bits = bits
        self.min_grid = -(2 ** (bits - 1))
        self.max_grid = 2 ** (bits - 1) - 1
        self.shift_values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=Q.dtype, device=self.device)
        self.idx = torch.arange(self.weight_shape[1], device=self.device) // groupsize

        scale_per_col = scales[:, self.idx]
        Q = Q / scale_per_col
        self.register_buffer("init", Q.to(self.dtype).detach())

        cand = self.init.unsqueeze(0) + self.shift_values.view(-1, 1, 1)
        self.register_buffer("valid_shift_mask", (cand >= self.min_grid) & (cand <= self.max_grid))

        logits = -0.5 * (torch.zeros_like(Q).unsqueeze(0) - self.shift_values.view(-1, 1, 1)) ** 2
        mean = (logits * self.valid_shift_mask.to(logits.dtype)).sum(dim=0, keepdim=True) / self.valid_shift_mask.sum(dim=0, keepdim=True)
        logits = logits - mean
        quant_logits = std * (torch.randn_like(logits) + logits * strength)

        self.quant_logits = nn.Parameter(quant_logits.to(self.logits_dtype).detach())
        self.scales = nn.Parameter(scales.float().detach())
        self.shift_values = self.shift_values.to(self.dtype)

    def forward(self, temperature, scale=1.0):
        return GumbelSoftmaxFunction.apply(
            self.quant_logits,
            self.scales,
            self.shift_values,
            self.valid_shift_mask,
            self.idx,
            self.init,
            float(temperature),
            float(scale),
            self.device
        )

    def get_hard_weights(self):
        masked_logits = self.quant_logits.masked_fill(~self.valid_shift_mask, -1e9)
        hard_shift_idx = torch.argmax(masked_logits, dim=0)
        scale_per_col = self.scales[:, self.idx].to(self.dtype)
        output = (self.init + self.shift_values[hard_shift_idx]) * scale_per_col

        return output, self.scales.to(self.dtype)

class GumbelSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quant_logits, scales, shift_values, valid_shift_mask, idx, init, temperature, scale, device):
        ctx.save_for_backward(quant_logits, scales)
        ctx.shift_values = shift_values
        ctx.valid_shift_mask = valid_shift_mask
        ctx.idx = idx
        ctx.init = init
        ctx.temperature = temperature
        ctx.scale = scale
        ctx.device = device

        ctx.cuda_fwd_rng_state = torch.cuda.get_rng_state(device=device)

        eps = 1e-8

        masked_logits = quant_logits.masked_fill(~valid_shift_mask, -1e9)
        u = torch.rand_like(masked_logits)
        noise = -torch.log(-torch.log(u + eps) + eps)
        soft_quant = F.softmax((masked_logits * ctx.scale + noise) / ctx.temperature, dim=0).to(init.dtype)
        soft_output = (soft_quant * shift_values.view(-1, 1, 1)).sum(dim=0)

        scale_per_col = scales[:, idx].to(soft_output.dtype)
        output = (init + soft_output) * scale_per_col

        return output

    @staticmethod
    def backward(ctx, grad_output):
        quant_logits, scales = ctx.saved_tensors
        shift_values = ctx.shift_values
        valid_shift_mask = ctx.valid_shift_mask
        idx = ctx.idx
        init = ctx.init
        temperature = ctx.temperature
        scale = ctx.scale
        device = ctx.device

        with torch.random.fork_rng(devices=[device]):
            torch.cuda.set_rng_state(ctx.cuda_fwd_rng_state, device=device)
            eps = 1e-8
            masked_logits = quant_logits.masked_fill(~valid_shift_mask, -1e9)
            u = torch.rand_like(masked_logits)
            noise = -torch.log(-torch.log(u + eps) + eps)
            soft_quant = F.softmax((masked_logits * ctx.scale + noise) / ctx.temperature, dim=0).to(init.dtype)
            soft_output = (soft_quant * shift_values.view(-1, 1, 1)).sum(dim=0)

        scale_per_col = scales[:, idx].to(soft_output.dtype)

        g_scale_per_col = grad_output * (init + soft_output)
        grad_scales = torch.zeros_like(scales)
        expand_idx = idx.unsqueeze(0).expand_as(g_scale_per_col)
        grad_scales.scatter_add_(1, expand_idx, g_scale_per_col.to(scales.dtype))

        g_soft_output = grad_output * scale_per_col
        g_q = g_soft_output.unsqueeze(0) * shift_values.view(-1, 1, 1)

        dot = (g_q * soft_quant).sum(dim=0, keepdim=True)
        grad_z = soft_quant * (g_q - dot)

        grad_quant_logits = grad_z * scale / temperature
        grad_quant_logits = grad_quant_logits.masked_fill(~valid_shift_mask, 0.0)

        return grad_quant_logits.float(), grad_scales, None, None, None, None, None, None, None
