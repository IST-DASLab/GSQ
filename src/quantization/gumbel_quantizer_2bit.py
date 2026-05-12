import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelQuantizer2Bit(nn.Module):
    def __init__(self, Q, scales, groupsize, std, strength, device, dtype, logits_dtype=None):
        super().__init__()
        self.weight_shape = tuple(Q.shape)
        self.device = device
        self.dtype = dtype
        self.logits_dtype = logits_dtype if logits_dtype is not None else dtype
        self.values = torch.tensor([-2, -1, 0, 1], dtype=Q.dtype, device=self.device)
        self.idx = torch.arange(self.weight_shape[1], device=self.device) // groupsize

        scale_per_col = scales[:, self.idx]
        Q = Q / scale_per_col

        logits = -0.5 * (Q.unsqueeze(0) - self.values.view(4, 1, 1)) ** 2
        logits = logits - logits.mean(dim=0, keepdim=True)
        quant_logits = std * (torch.randn((4, Q.shape[0], Q.shape[1]), dtype=Q.dtype, device=self.device) + logits * strength)

        self.values = self.values.to(self.dtype)

        self.quant_logits = nn.Parameter(quant_logits.to(self.logits_dtype).detach())
        self.scales = nn.Parameter(scales.float().detach())

    def forward(self, temperature, scale=1.0):
        return GumbelSoftmaxFunction.apply(
            self.quant_logits,
            self.scales,
            self.values,
            self.idx,
            float(temperature),
            float(scale),
            self.device,
        )

    def get_hard_weights(self):
        hard_mask_idx = torch.argmax(self.quant_logits, dim=0)
        scale_per_col = self.scales[:, self.idx].to(self.dtype)
        output = self.values[hard_mask_idx] * scale_per_col
        return output, self.scales.to(self.dtype)


class GumbelSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quant_logits, scales, values, idx, temperature, scale, device):
        ctx.save_for_backward(quant_logits, scales)
        ctx.values = values
        ctx.idx = idx
        ctx.temperature = temperature
        ctx.scale = scale
        ctx.device = device

        ctx.cuda_fwd_rng_state = torch.cuda.get_rng_state(device=device)

        eps = 1e-10

        u = torch.rand_like(quant_logits)
        noise = -torch.log(-torch.log(u + eps) + eps).to(values.dtype)
        soft_quant = F.softmax((quant_logits.to(values.dtype) * ctx.scale + noise) / ctx.temperature, dim=0)
        soft_output = (soft_quant * values.view(4, 1, 1)).sum(dim=0)

        scale_per_col = scales[:, idx].to(values.dtype)
        output = soft_output * scale_per_col

        return output

    @staticmethod
    def backward(ctx, grad_output):
        quant_logits, scales = ctx.saved_tensors
        values = ctx.values
        idx = ctx.idx
        temperature = ctx.temperature
        scale = ctx.scale
        device = ctx.device

        with torch.random.fork_rng(devices=[device]):
            torch.cuda.set_rng_state(ctx.cuda_fwd_rng_state, device=device)
            eps = 1e-10
            u = torch.rand_like(quant_logits)
            noise = -torch.log(-torch.log(u + eps) + eps).to(values.dtype)
            soft_quant = F.softmax((quant_logits.to(values.dtype) * scale + noise) / temperature, dim=0)
            soft_output = (soft_quant * values.view(4, 1, 1)).sum(dim=0)

        scale_per_col = scales[:, idx].to(values.dtype)

        g_scale_per_col = grad_output * soft_output
        grad_scales = torch.zeros_like(scales)
        expand_idx = idx.unsqueeze(0).expand_as(g_scale_per_col)
        grad_scales.scatter_add_(1, expand_idx, g_scale_per_col.float())

        g_soft_output = grad_output * scale_per_col

        g_q = g_soft_output.unsqueeze(0) * values.view(4, 1, 1)

        dot = (g_q * soft_quant).sum(dim=0, keepdim=True)
        grad_z = soft_quant * (g_q - dot)

        grad_quant_logits = grad_z * scale / temperature

        return grad_quant_logits.to(quant_logits.dtype), grad_scales, None, None, None, None, None
