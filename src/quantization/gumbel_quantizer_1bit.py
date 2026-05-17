import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelQuantizer1Bit(nn.Module):
    def __init__(self, Q, scales, groupsize, std, strength, device, dtype, logits_dtype=None):
        super().__init__()
        self.weight_shape = tuple(Q.shape)
        self.device = device
        self.dtype = dtype
        self.logits_dtype = logits_dtype if logits_dtype is not None else dtype
        self.idx = torch.arange(self.weight_shape[1], device=self.device) // groupsize

        scale_per_col = scales[:, self.idx]
        Q = Q / scale_per_col

        sign_logits = (std * (torch.randn_like(Q) + Q * strength)).to(self.dtype)

        self.sign_logits = nn.Parameter(sign_logits.to(self.logits_dtype).detach())
        self.scales = nn.Parameter(scales.float().detach())

    def forward(self, temperature, scale=1.0):
        return GumbelSoftmaxFunction.apply(
            self.sign_logits,
            self.scales,
            self.idx,
            float(temperature),
            float(scale),
            self.device,
            self.dtype
        )

    def get_hard_weights(self):
        hard_mask = (self.sign_logits > 0).to(self.dtype)
        hard_sign = 2.0 * hard_mask - 1.0
        scale_per_col = self.scales[:, self.idx].to(self.dtype)
        output = hard_sign * scale_per_col

        return output, self.scales.to(self.dtype)


class GumbelSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sign_logits, scales, idx, temperature, scale, device, dtype):
        ctx.save_for_backward(sign_logits, scales)
        ctx.idx = idx
        ctx.temperature = temperature
        ctx.scale = scale
        ctx.device = device
        ctx.dtype = dtype

        ctx.cuda_fwd_rng_state = torch.cuda.get_rng_state(device=device)

        eps = 1e-8
        u = torch.rand_like(sign_logits)
        noise = torch.logit(u, eps=eps)
        soft_sign = (2.0 * torch.sigmoid((2.0 * sign_logits * scale + noise) / temperature) - 1.0).to(dtype)

        scale_per_col = scales[:, idx].to(dtype)

        output = soft_sign * scale_per_col

        return output

    @staticmethod
    def backward(ctx, grad_output):
        sign_logits, scales = ctx.saved_tensors
        idx = ctx.idx
        temperature = ctx.temperature
        scale = ctx.scale
        device = ctx.device
        dtype = ctx.dtype

        with torch.random.fork_rng(devices=[device]):
            torch.cuda.set_rng_state(ctx.cuda_fwd_rng_state, device=device)
            eps = 1e-8
            u = torch.rand_like(sign_logits)
            noise = torch.logit(u, eps=eps)
            soft_sign = (2.0 * torch.sigmoid((2.0 * sign_logits * scale + noise) / temperature) - 1.0).to(dtype)

        scale_per_col = scales[:, idx].to(dtype)

        grad_soft_quant = grad_output * scale_per_col
        grad_scale_per_col = grad_output * soft_sign

        grad_quant_logits = grad_soft_quant * (1.0 - soft_sign.pow(2)) * (2.0 * scale / temperature)

        grad_scales = torch.zeros_like(scales)
        idx_expanded = idx.unsqueeze(0).expand(grad_scales.size(0), -1)
        grad_scales.scatter_add_(1, idx_expanded, grad_scale_per_col.float())

        return grad_quant_logits.to(sign_logits.dtype), grad_scales, None, None, None, None, None
