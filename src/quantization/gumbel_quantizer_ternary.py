import math

import torch
import torch.nn as nn

from .gumbel_utils import fork_rng_with_state, get_rng_state


_STANDARD_MASK_MODE = "standard"
_FIXED_DENSITY_MASK_MODE = "fixed_density"


def _canonical_scope(scope):
    if scope in ("layer", "tensor"):
        return "tensor"
    if scope in ("group", "row_group", "quant_group"):
        return "group"
    if scope == "row":
        return "row"
    raise ValueError(
        f"Unsupported ternary_density_scope={scope!r}. "
        "Supported: 'row', 'group' (aliases: 'row_group', 'quant_group'), 'tensor' (alias: 'layer')."
    )


def _validate_density(density):
    density = float(density)
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"ternary_density must be in [0, 1], got {density}")
    return density


def _aln_components(mask_scores, idx, density, scope, eps):
    scores = mask_scores.float()
    abs_scores = scores.abs() + eps
    sign_scores = torch.sign(scores)
    rows, columns = scores.shape

    if scope == "tensor":
        z = abs_scores.sum().clamp_min(eps)
        pi = abs_scores / z
        k = math.floor(density * rows * columns)
        k_tensor = torch.tensor(float(k), dtype=torch.float32, device=scores.device)
        p_raw = k_tensor * pi
        return p_raw, pi, sign_scores, z, k_tensor

    if scope == "row":
        z = abs_scores.sum(dim=1, keepdim=True).clamp_min(eps)
        pi = abs_scores / z
        k = math.floor(density * columns)
        k_tensor = torch.full((rows, 1), float(k), dtype=torch.float32, device=scores.device)
        p_raw = k_tensor * pi
        return p_raw, pi, sign_scores, z, k_tensor

    n_groups = int(idx.max().item()) + 1
    idx_expanded = idx.unsqueeze(0).expand(rows, -1)
    z = torch.zeros(rows, n_groups, dtype=torch.float32, device=scores.device)
    z.scatter_add_(1, idx_expanded, abs_scores)
    z = z.clamp_min(eps)

    group_counts = torch.bincount(idx, minlength=n_groups).to(torch.float32).to(scores.device)
    k_by_group = torch.floor(group_counts * density)
    z_per_col = z[:, idx]
    k_per_col = k_by_group[idx].unsqueeze(0)
    pi = abs_scores / z_per_col
    p_raw = k_per_col * pi
    return p_raw, pi, sign_scores, z_per_col, k_per_col


def _aln_probabilities(mask_scores, idx, density, scope, eps):
    p_raw, _, _, _, _ = _aln_components(mask_scores, idx, density, scope, eps)
    p_clipped = p_raw.clamp(eps, 1.0 - eps)
    return p_raw + (p_clipped - p_raw).detach()


def _aln_score_backward(grad_p, mask_scores, idx, density, scope, eps):
    p_raw, pi, sign_scores, z, k = _aln_components(mask_scores, idx, density, scope, eps)
    del p_raw

    grad_p = grad_p.float()
    if scope == "tensor":
        baseline = (pi * grad_p).sum()
        return k * sign_scores / z * (grad_p - baseline)

    if scope == "row":
        baseline = (pi * grad_p).sum(dim=1, keepdim=True)
        return k * sign_scores / z * (grad_p - baseline)

    rows = mask_scores.shape[0]
    n_groups = int(idx.max().item()) + 1
    idx_expanded = idx.unsqueeze(0).expand(rows, -1)
    baseline = torch.zeros(rows, n_groups, dtype=torch.float32, device=mask_scores.device)
    baseline.scatter_add_(1, idx_expanded, pi * grad_p)
    baseline_per_col = baseline[:, idx]
    return k * sign_scores / z * (grad_p - baseline_per_col)


def _topk_mask(mask_scores, idx, density, scope):
    scores = mask_scores.abs()
    rows, columns = scores.shape
    hard_mask = torch.zeros_like(scores, dtype=torch.bool)

    if scope == "tensor":
        k = math.floor(density * rows * columns)
        if k <= 0:
            return hard_mask
        if k >= rows * columns:
            return torch.ones_like(hard_mask)
        top_idx = torch.topk(scores.flatten(), k, largest=True).indices
        hard_mask.view(-1).scatter_(0, top_idx, True)
        return hard_mask

    if scope == "row":
        k = math.floor(density * columns)
        if k <= 0:
            return hard_mask
        if k >= columns:
            return torch.ones_like(hard_mask)
        top_idx = torch.topk(scores, k, dim=1, largest=True).indices
        hard_mask.scatter_(1, top_idx, True)
        return hard_mask

    n_groups = int(idx.max().item()) + 1
    for group_idx in range(n_groups):
        cols = torch.nonzero(idx == group_idx, as_tuple=True)[0]
        k = math.floor(density * cols.numel())
        if k <= 0:
            continue
        if k >= cols.numel():
            hard_mask[:, cols] = True
            continue
        top_local = torch.topk(scores[:, cols], k, dim=1, largest=True).indices
        selected_cols = cols[top_local]
        hard_mask.scatter_(1, selected_cols, True)
    return hard_mask


class GumbelQuantizerTernary(nn.Module):
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
        mask_mode=_STANDARD_MASK_MODE,
        density=0.5,
        density_scope="row",
        density_eps=1e-6,
    ):
        super().__init__()
        self.weight_shape = tuple(Q.shape)
        self.device = torch.device(device)
        self.dtype = dtype
        self.logits_dtype = logits_dtype if logits_dtype is not None else dtype
        self.idx = torch.arange(self.weight_shape[1], device=self.device) // groupsize
        self.mask_mode = mask_mode
        self.density = _validate_density(density)
        self.density_scope = _canonical_scope(density_scope)
        self.density_eps = float(density_eps)
        if self.density_eps <= 0.0 or self.density_eps >= 0.5:
            raise ValueError(f"ternary_density_eps must be in (0, 0.5), got {self.density_eps}")

        if self.mask_mode not in (_STANDARD_MASK_MODE, _FIXED_DENSITY_MASK_MODE):
            raise ValueError(
                f"Unsupported ternary_mask_mode={self.mask_mode!r}. "
                "Supported: 'standard', 'fixed_density'."
            )

        sign_logits = (std * (torch.randn_like(Q) + torch.sign(Q) * strength)).to(self.dtype)

        scale_per_col = scales[:, self.idx].to(Q.dtype)
        scale_per_col = scale_per_col.abs().clamp_min(torch.finfo(scale_per_col.dtype).eps)
        normalized_abs_q = (Q / scale_per_col).abs()

        if self.mask_mode == _FIXED_DENSITY_MASK_MODE:
            mask_logits = std * (
                1.0 + normalized_abs_q * strength + 0.01 * torch.randn_like(Q)
            )
            self.no_weight_decay_param_names = ("mask_logits",)
        else:
            mask_logits = (std * (torch.randn_like(Q) + (2 * normalized_abs_q - 1) * strength)).to(self.dtype)
            self.no_weight_decay_param_names = ()

        self.sign_logits = nn.Parameter(sign_logits.to(self.logits_dtype).detach())
        self.mask_logits = nn.Parameter(mask_logits.to(self.logits_dtype).detach())
        self.scales = nn.Parameter(scales.float().clone().detach())

    def forward(self, temperature, scale=1.0):
        if self.mask_mode == _FIXED_DENSITY_MASK_MODE:
            return FixedDensityTernaryFunction.apply(
                self.sign_logits,
                self.mask_logits,
                self.scales,
                self.idx,
                self.density,
                self.density_scope,
                self.density_eps,
                float(temperature),
                float(scale),
                self.device,
                self.dtype,
            )

        return GumbelSoftmaxFunction.apply(
            self.sign_logits,
            self.mask_logits,
            self.scales,
            self.idx,
            float(temperature),
            float(scale),
            self.device,
            self.dtype,
        )

    def get_hard_weights(self):
        if self.mask_mode == _FIXED_DENSITY_MASK_MODE:
            hard_mask = _topk_mask(
                self.mask_logits.detach(),
                self.idx,
                self.density,
                self.density_scope,
            ).to(self.dtype)
        else:
            hard_mask = (self.mask_logits > 0).to(self.dtype)

        hard_sign = torch.where(
            self.sign_logits >= 0,
            torch.ones_like(self.sign_logits, dtype=self.dtype),
            -torch.ones_like(self.sign_logits, dtype=self.dtype),
        )
        scale_per_col = self.scales[:, self.idx].to(self.dtype)
        output = hard_mask * hard_sign * scale_per_col

        return output, self.scales.to(self.dtype)


class GumbelSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sign_logits, mask_logits, scales, idx, temperature, scale, device, dtype):
        ctx.save_for_backward(sign_logits, mask_logits, scales)
        ctx.idx = idx
        ctx.temperature = temperature
        ctx.scale = scale
        ctx.device = torch.device(device)
        ctx.dtype = dtype
        ctx.fwd_rng_state = get_rng_state(ctx.device)

        eps = 1e-8
        u_mask = torch.rand_like(mask_logits)
        mask_noise = torch.logit(u_mask, eps=eps).to(ctx.dtype)
        soft_mask = torch.sigmoid((2.0 * mask_logits.to(ctx.dtype) * ctx.scale + mask_noise) / ctx.temperature)

        u_sign = torch.rand_like(sign_logits)
        sign_noise = torch.logit(u_sign, eps=eps).to(ctx.dtype)
        soft_sign = 2.0 * torch.sigmoid((2.0 * sign_logits.to(ctx.dtype) * ctx.scale + sign_noise) / ctx.temperature) - 1.0

        scale_per_col = scales[:, idx].to(ctx.dtype)
        output = soft_mask * soft_sign * scale_per_col

        return output

    @staticmethod
    def backward(ctx, grad_output):
        sign_logits, mask_logits, scales = ctx.saved_tensors
        idx = ctx.idx
        temperature = ctx.temperature
        scale = ctx.scale

        with fork_rng_with_state(ctx.device, ctx.fwd_rng_state):
            eps = 1e-8
            u_mask = torch.rand_like(mask_logits)
            mask_noise = torch.logit(u_mask, eps=eps).to(ctx.dtype)
            soft_mask = torch.sigmoid((2.0 * mask_logits.to(ctx.dtype) * scale + mask_noise) / temperature)

            u_sign = torch.rand_like(sign_logits)
            sign_noise = torch.logit(u_sign, eps=eps).to(ctx.dtype)
            soft_sign = 2.0 * torch.sigmoid((2.0 * sign_logits.to(ctx.dtype) * scale + sign_noise) / temperature) - 1.0

        scale_per_col = scales[:, idx].to(ctx.dtype)

        grad_soft_mask = grad_output * soft_sign * scale_per_col
        grad_soft_sign = grad_output * soft_mask * scale_per_col
        grad_scale_per_col = grad_output * soft_mask * soft_sign

        grad_mask_logits = grad_soft_mask * soft_mask * (1.0 - soft_mask) * (2.0 * scale / temperature)
        grad_sign_logits = grad_soft_sign * (1.0 - soft_sign.pow(2)) * scale / temperature

        grad_scales = torch.zeros_like(scales)
        idx_expanded = idx.unsqueeze(0).expand(grad_scales.size(0), -1)
        grad_scales.scatter_add_(1, idx_expanded, grad_scale_per_col.float())

        return grad_sign_logits.to(sign_logits.dtype), grad_mask_logits.to(mask_logits.dtype), grad_scales, None, None, None, None, None


class FixedDensityTernaryFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        sign_logits,
        mask_scores,
        scales,
        idx,
        density,
        density_scope,
        density_eps,
        temperature,
        scale,
        device,
        dtype,
    ):
        ctx.save_for_backward(sign_logits, mask_scores, scales)
        ctx.idx = idx
        ctx.density = float(density)
        ctx.density_scope = density_scope
        ctx.density_eps = float(density_eps)
        ctx.temperature = temperature
        ctx.scale = scale
        ctx.device = torch.device(device)
        ctx.dtype = dtype
        ctx.fwd_rng_state = get_rng_state(ctx.device)

        eps = 1e-8
        mask_prob = _aln_probabilities(mask_scores, idx, ctx.density, density_scope, ctx.density_eps)
        u_mask = torch.rand(mask_scores.shape, dtype=torch.float32, device=mask_scores.device)
        mask_noise = torch.logit(u_mask, eps=eps)
        soft_mask = torch.sigmoid((torch.logit(mask_prob, eps=ctx.density_eps) + mask_noise) / ctx.temperature)

        u_sign = torch.rand(sign_logits.shape, dtype=torch.float32, device=sign_logits.device)
        sign_noise = torch.logit(u_sign, eps=eps)
        soft_sign = 2.0 * torch.sigmoid((2.0 * sign_logits.float() * ctx.scale + sign_noise) / ctx.temperature) - 1.0

        scale_per_col = scales[:, idx].float()
        output = soft_mask * soft_sign * scale_per_col
        return output.to(ctx.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        sign_logits, mask_scores, scales = ctx.saved_tensors
        idx = ctx.idx

        with fork_rng_with_state(ctx.device, ctx.fwd_rng_state):
            eps = 1e-8
            mask_prob = _aln_probabilities(
                mask_scores,
                idx,
                ctx.density,
                ctx.density_scope,
                ctx.density_eps,
            )
            u_mask = torch.rand(mask_scores.shape, dtype=torch.float32, device=mask_scores.device)
            mask_noise = torch.logit(u_mask, eps=eps)
            soft_mask = torch.sigmoid((torch.logit(mask_prob, eps=ctx.density_eps) + mask_noise) / ctx.temperature)

            u_sign = torch.rand(sign_logits.shape, dtype=torch.float32, device=sign_logits.device)
            sign_noise = torch.logit(u_sign, eps=eps)
            soft_sign = 2.0 * torch.sigmoid((2.0 * sign_logits.float() * ctx.scale + sign_noise) / ctx.temperature) - 1.0

        grad_output = grad_output.float()
        scale_per_col = scales[:, idx].float()

        grad_soft_mask = grad_output * soft_sign * scale_per_col
        grad_soft_sign = grad_output * soft_mask * scale_per_col
        grad_scale_per_col = grad_output * soft_mask * soft_sign

        grad_p = (
            grad_soft_mask
            * soft_mask
            * (1.0 - soft_mask)
            / ctx.temperature
            / (mask_prob * (1.0 - mask_prob)).clamp_min(ctx.density_eps)
        )
        grad_mask_scores = _aln_score_backward(
            grad_p,
            mask_scores,
            idx,
            ctx.density,
            ctx.density_scope,
            ctx.density_eps,
        )
        grad_sign_logits = grad_soft_sign * (1.0 - soft_sign.pow(2)) * ctx.scale / ctx.temperature

        grad_scales = torch.zeros_like(scales)
        idx_expanded = idx.unsqueeze(0).expand(grad_scales.size(0), -1)
        grad_scales.scatter_add_(1, idx_expanded, grad_scale_per_col.to(scales.dtype))

        return (
            grad_sign_logits.to(sign_logits.dtype),
            grad_mask_scores.to(mask_scores.dtype),
            grad_scales,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
