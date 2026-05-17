"""Binary GSQ quantizers.

This module contains the standard one-logit binary GSQ path and two ALN
ablation modes. The ALN modes keep the binary grid fixed at {-scale, +scale},
but replace the sign logit with a local two-candidate absolute-score
parameterization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gumbel_utils import fork_rng_with_state, get_rng_state


_STANDARD_MODE = "standard"
_ALN_MODE = "aln"
_ALN_ST_MODE = "aln_st"


def _canonical_binary_mode(mode):
    mode = str(mode).lower()
    if mode in ("standard", "gumbel", "gumbel_sigmoid"):
        return _STANDARD_MODE
    if mode in ("aln", "aln_softmax", "aln_gumbel"):
        return _ALN_MODE
    if mode in ("aln_st", "aln_ste", "aln_backward", "aln_backwards"):
        return _ALN_ST_MODE
    raise ValueError(
        f"Unsupported binary_mode={mode!r}. "
        "Supported: 'standard', 'aln', 'aln_st' (aliases: 'aln_backward')."
    )


def _aln_binary_probabilities(sign_scores, eps):
    # ALN uses absolute scores as local simplex mass, so multiplying all scores
    # by a constant does not change the forward probability.
    abs_scores = sign_scores.float().abs() + eps
    z = abs_scores.sum(dim=0, keepdim=True).clamp_min(eps)
    return abs_scores / z


def _aln_binary_score_backward(grad_prob, sign_scores, prob, eps):
    # Advantage-style ALN gradient: each sign candidate is compared with the
    # current probability-weighted average utility for that weight coordinate.
    abs_sum = (sign_scores.float().abs() + eps).sum(dim=0, keepdim=True).clamp_min(eps)
    baseline = (prob * grad_prob).sum(dim=0, keepdim=True)
    return torch.sign(sign_scores.float()) / abs_sum * (grad_prob - baseline)


class GumbelQuantizer1Bit(nn.Module):
    """One-bit GSQ with standard and ALN sign-assignment variants."""

    def __init__(
        self,
        Q,
        scales,
        groupsize,
        std,
        strength,
        device,
        dtype,
        logits_dtype=None,
        binary_mode=_STANDARD_MODE,
        aln_eps=1e-6,
    ):
        super().__init__()
        self.weight_shape = tuple(Q.shape)
        self.device = torch.device(device)
        self.dtype = dtype
        self.logits_dtype = logits_dtype if logits_dtype is not None else dtype
        self.idx = torch.arange(self.weight_shape[1], device=self.device) // groupsize
        self.binary_mode = _canonical_binary_mode(binary_mode)
        self.aln_eps = float(aln_eps)
        if self.aln_eps <= 0.0 or self.aln_eps >= 0.5:
            raise ValueError(f"binary_aln_eps must be in (0, 0.5), got {self.aln_eps}")

        scale_per_col = scales[:, self.idx].abs().clamp_min(torch.finfo(scales.dtype).eps)
        normalized_q = Q / scale_per_col
        sign_prior = torch.where(normalized_q >= 0, 1.0, -1.0)

        if self.binary_mode == _STANDARD_MODE:
            sign_logits = std * (torch.randn_like(Q) + sign_prior * strength)
            self.sign_logits = nn.Parameter(sign_logits.to(self.logits_dtype).detach())
            self.no_weight_decay_param_names = ()
        else:
            # Store separate absolute scores for {-1, +1}. Weight decay is
            # disabled for these scores because ALN probabilities are
            # scale-invariant but their gradient magnitude is not.
            neg_prior = (sign_prior < 0).to(Q.dtype)
            pos_prior = (sign_prior >= 0).to(Q.dtype)
            sign_scores = torch.stack(
                [
                    1.0 + strength * neg_prior + 0.01 * torch.randn_like(Q),
                    1.0 + strength * pos_prior + 0.01 * torch.randn_like(Q),
                ],
                dim=0,
            )
            self.sign_scores = nn.Parameter(sign_scores.to(self.logits_dtype).detach())
            self.no_weight_decay_param_names = ("sign_scores",)

        self.scales = nn.Parameter(scales.float().clone().detach())

    def forward(self, temperature, scale=1.0):
        if self.binary_mode in (_ALN_MODE, _ALN_ST_MODE):
            return ALNBinarySignFunction.apply(
                self.sign_scores,
                self.scales,
                self.idx,
                self.binary_mode == _ALN_ST_MODE,
                self.aln_eps,
                float(temperature),
                float(scale),
                self.device,
                self.dtype,
            )

        return GumbelBinarySignFunction.apply(
            self.sign_logits,
            self.scales,
            self.idx,
            float(temperature),
            float(scale),
            self.device,
            self.dtype,
        )

    def get_hard_weights(self):
        if self.binary_mode in (_ALN_MODE, _ALN_ST_MODE):
            hard_idx = torch.argmax(self.sign_scores.abs(), dim=0)
            hard_sign = torch.where(
                hard_idx == 1,
                torch.ones_like(hard_idx, dtype=self.dtype),
                -torch.ones_like(hard_idx, dtype=self.dtype),
            )
        else:
            hard_sign = torch.where(
                self.sign_logits >= 0,
                torch.ones_like(self.sign_logits, dtype=self.dtype),
                -torch.ones_like(self.sign_logits, dtype=self.dtype),
            )
        scale_per_col = self.scales[:, self.idx].to(self.dtype)
        output = hard_sign * scale_per_col
        return output, self.scales.to(self.dtype)


class GumbelBinarySignFunction(torch.autograd.Function):
    """Binary Concrete relaxation for the standard one-logit GSQ path."""

    @staticmethod
    def forward(ctx, sign_logits, scales, idx, temperature, scale, device, dtype):
        ctx.save_for_backward(sign_logits, scales)
        ctx.idx = idx
        ctx.temperature = temperature
        ctx.scale = scale
        ctx.device = torch.device(device)
        ctx.dtype = dtype
        ctx.fwd_rng_state = get_rng_state(ctx.device)

        eps = 1e-8
        u_sign = torch.rand_like(sign_logits)
        sign_noise = torch.logit(u_sign, eps=eps).to(ctx.dtype)
        soft_sign = 2.0 * torch.sigmoid(
            (2.0 * sign_logits.to(ctx.dtype) * ctx.scale + sign_noise) / ctx.temperature
        ) - 1.0

        scale_per_col = scales[:, idx].to(ctx.dtype)
        output = soft_sign * scale_per_col
        return output

    @staticmethod
    def backward(ctx, grad_output):
        sign_logits, scales = ctx.saved_tensors
        idx = ctx.idx
        temperature = ctx.temperature
        scale = ctx.scale

        with fork_rng_with_state(ctx.device, ctx.fwd_rng_state):
            eps = 1e-8
            u_sign = torch.rand_like(sign_logits)
            sign_noise = torch.logit(u_sign, eps=eps).to(ctx.dtype)
            soft_sign = 2.0 * torch.sigmoid(
                (2.0 * sign_logits.to(ctx.dtype) * scale + sign_noise) / temperature
            ) - 1.0

        scale_per_col = scales[:, idx].to(ctx.dtype)

        grad_soft_sign = grad_output * scale_per_col
        grad_scale_per_col = grad_output * soft_sign

        grad_sign_logits = grad_soft_sign * (1.0 - soft_sign.pow(2)) * scale / temperature

        grad_scales = torch.zeros_like(scales)
        idx_expanded = idx.unsqueeze(0).expand(grad_scales.size(0), -1)
        grad_scales.scatter_add_(1, idx_expanded, grad_scale_per_col.float())

        return grad_sign_logits.to(sign_logits.dtype), grad_scales, None, None, None, None, None


class ALNBinarySignFunction(torch.autograd.Function):
    """Gumbel-Softmax over ALN sign probabilities.

    ``aln`` differentiates through the sampled softmax probabilities, while
    ``aln_st`` uses the biased ALN-backward surrogate over the same forward
    sample.
    """

    @staticmethod
    def forward(ctx, sign_scores, scales, idx, straight_through, aln_eps, temperature, scale, device, dtype):
        del scale
        ctx.save_for_backward(sign_scores, scales)
        ctx.idx = idx
        ctx.straight_through = bool(straight_through)
        ctx.aln_eps = float(aln_eps)
        ctx.temperature = temperature
        ctx.device = torch.device(device)
        ctx.dtype = dtype
        ctx.fwd_rng_state = get_rng_state(ctx.device)

        prob = _aln_binary_probabilities(sign_scores, ctx.aln_eps)
        eps = 1e-8
        u = torch.rand(sign_scores.shape, dtype=torch.float32, device=sign_scores.device)
        noise = -torch.log(-torch.log(u + eps) + eps)
        soft_sign_prob = F.softmax((torch.log(prob.clamp_min(ctx.aln_eps)) + noise) / ctx.temperature, dim=0)
        soft_sign = soft_sign_prob[1] - soft_sign_prob[0]

        scale_per_col = scales[:, idx].float()
        output = soft_sign * scale_per_col
        return output.to(ctx.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        sign_scores, scales = ctx.saved_tensors
        idx = ctx.idx

        with fork_rng_with_state(ctx.device, ctx.fwd_rng_state):
            prob = _aln_binary_probabilities(sign_scores, ctx.aln_eps)
            eps = 1e-8
            u = torch.rand(sign_scores.shape, dtype=torch.float32, device=sign_scores.device)
            noise = -torch.log(-torch.log(u + eps) + eps)
            soft_sign_prob = F.softmax((torch.log(prob.clamp_min(ctx.aln_eps)) + noise) / ctx.temperature, dim=0)
            soft_sign = soft_sign_prob[1] - soft_sign_prob[0]

        grad_output = grad_output.float()
        scale_per_col = scales[:, idx].float()
        grad_soft_sign = grad_output * scale_per_col
        grad_scale_per_col = grad_output * soft_sign

        values = torch.tensor([-1.0, 1.0], dtype=torch.float32, device=sign_scores.device).view(2, 1, 1)
        grad_y = grad_soft_sign.unsqueeze(0) * values
        if ctx.straight_through:
            # Straight-through ALN-backward: forward uses the sampled
            # Gumbel-Softmax value, but backward treats y as the local simplex.
            grad_prob = grad_y
        else:
            dot = (grad_y * soft_sign_prob).sum(dim=0, keepdim=True)
            grad_z = soft_sign_prob * (grad_y - dot)
            grad_prob = grad_z / ctx.temperature / prob.clamp_min(ctx.aln_eps)

        grad_sign_scores = _aln_binary_score_backward(grad_prob, sign_scores, prob, ctx.aln_eps)

        grad_scales = torch.zeros_like(scales)
        idx_expanded = idx.unsqueeze(0).expand(grad_scales.size(0), -1)
        grad_scales.scatter_add_(1, idx_expanded, grad_scale_per_col.to(scales.dtype))

        return (
            grad_sign_scores.to(sign_scores.dtype),
            grad_scales,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
