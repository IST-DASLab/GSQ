from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_IQUANT_GRIDS = {
    "IQ1_S",
    "IQ1_M",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ3_XXS",
    "IQ3_S",
}


def _as_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def all_sign_patterns(dim: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Return every +/-1 sign pattern for a small vector dimension."""
    device = _as_device(device)
    if dim <= 0 or dim > 16:
        raise ValueError(f"Expected 1 <= dim <= 16, got {dim}")
    values = torch.arange(2**dim, device=device, dtype=torch.long)
    bits = (values[:, None] >> torch.arange(dim, device=device, dtype=torch.long)) & 1
    return torch.where(bits == 0, torch.ones((), device=device), -torch.ones((), device=device)).float()


def load_iquant_magnitude_codebook(
    qtype_name: str,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load the small IQuant magnitude grid from gguf.quants.

    The returned rows are the raw grid vectors used by llama.cpp's Python GGUF
    quant helpers. IQuant scales are intentionally kept outside the codebook so
    GSVQ can freeze them in the initial implementation.
    """
    qtype_name = qtype_name.upper()
    if qtype_name not in SUPPORTED_IQUANT_GRIDS:
        raise ValueError(
            f"Unsupported IQuant grid {qtype_name!r}. "
            f"Supported: {sorted(SUPPORTED_IQUANT_GRIDS)}"
        )

    try:
        import gguf.quants as gguf_quants
    except ImportError as exc:
        raise ImportError("GSVQ IQuant codebooks require the `gguf` Python package") from exc

    cls = getattr(gguf_quants, qtype_name)
    cls.init_grid()
    if cls.grid is None:
        raise ValueError(f"{qtype_name} does not expose a vector grid")

    grid = torch.from_numpy(cls.grid.reshape(-1, cls.grid_shape[-1]).copy())
    return grid.to(device=_as_device(device), dtype=dtype)


def _weighted_sqdist(
    x: torch.Tensor,
    codebook: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    diff = x[:, None, :] - codebook[None, :, :]
    dist = diff.square()
    if weights is not None:
        dist = dist * weights[:, None, :]
    return dist.sum(dim=-1)


def nearest_code_indices(
    vectors: torch.Tensor,
    codebook: torch.Tensor,
    *,
    topk: int = 1,
    weights: Optional[torch.Tensor] = None,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Find top-k nearest codebook rows for each vector without materializing all rows."""
    if vectors.ndim != 2 or codebook.ndim != 2:
        raise ValueError("vectors and codebook must be rank-2 tensors")
    if vectors.shape[-1] != codebook.shape[-1]:
        raise ValueError(
            f"Vector/codebook dims differ: {vectors.shape[-1]} vs {codebook.shape[-1]}"
        )
    topk = max(1, min(int(topk), codebook.shape[0]))
    out = []
    for start in range(0, vectors.shape[0], chunk_size):
        end = min(start + chunk_size, vectors.shape[0])
        w = None if weights is None else weights[start:end]
        dist = _weighted_sqdist(vectors[start:end].float(), codebook.float(), w)
        out.append(torch.topk(dist, k=topk, dim=-1, largest=False).indices)
    return torch.cat(out, dim=0)


def codebook_neighbor_indices(
    codebook: torch.Tensor,
    *,
    topk: int,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Precompute top-k codebook neighbors for shift-style candidate sets."""
    topk = max(1, min(int(topk), codebook.shape[0]))
    if weights is not None:
        # A single static neighbor table only makes sense for shared weights.
        if weights.ndim != 1:
            raise ValueError("codebook neighbor weights must be a rank-1 tensor")
        dist = _weighted_sqdist(codebook.float(), codebook.float(), weights.view(1, -1))
    else:
        dist = _weighted_sqdist(codebook.float(), codebook.float())
    return torch.topk(dist, k=topk, dim=-1, largest=False).indices


def build_candidate_indices(
    target_vectors: torch.Tensor,
    codebook: torch.Tensor,
    init_indices: torch.Tensor,
    *,
    candidate_count: int,
    neighbor_candidates: int,
    target_candidates: int,
    weights: Optional[torch.Tensor] = None,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Build a fixed candidate set per vector.

    The set is the union of (a) nearby codebook rows around the current code,
    which is the VQ analogue of GSQ's shift idea, and (b) nearest rows to the
    current full-precision target, which keeps the search from getting trapped
    when the initial IQuant code is poor.
    """
    if init_indices.ndim != 1:
        init_indices = init_indices.reshape(-1)
    if init_indices.shape[0] != target_vectors.shape[0]:
        raise ValueError("init_indices must have one entry per target vector")

    candidate_count = int(candidate_count)
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")

    parts = [init_indices[:, None].to(torch.long)]

    if neighbor_candidates > 1:
        neighbors = codebook_neighbor_indices(codebook, topk=neighbor_candidates)
        parts.append(neighbors[init_indices.to(torch.long)])

    if target_candidates > 0:
        target_nn = nearest_code_indices(
            target_vectors,
            codebook,
            topk=target_candidates,
            weights=weights,
            chunk_size=chunk_size,
        )
        parts.append(target_nn)

    raw = torch.cat(parts, dim=1)
    if raw.shape[1] >= candidate_count:
        return raw[:, :candidate_count].contiguous()

    filler_width = candidate_count - raw.shape[1]
    filler = torch.arange(filler_width, dtype=torch.long, device=target_vectors.device)
    filler = filler.remainder(codebook.shape[0]).view(1, -1).expand(target_vectors.shape[0], -1)
    return torch.cat([raw, filler], dim=1).contiguous()


def _rotate_grad_between(
    grad: torch.Tensor,
    from_vec: torch.Tensor,
    to_vec: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    from_norm = from_vec.norm(dim=-1, keepdim=True).clamp_min(eps)
    to_norm = to_vec.norm(dim=-1, keepdim=True).clamp_min(eps)
    a = from_vec / from_norm
    b = to_vec / to_norm
    cos = (a * b).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    sin = torch.sqrt((1.0 - cos.square()).clamp_min(eps))
    c = (b - cos * a) / sin

    g_a = (grad * a).sum(dim=-1, keepdim=True)
    g_c = (grad * c).sum(dim=-1, keepdim=True)
    rest = grad - g_a * a - g_c * c
    rotated = (g_a * cos - g_c * sin) * a + (g_a * sin + g_c * cos) * c + rest
    return rotated


class RotationGradient(torch.autograd.Function):
    """Gradient-only rotation preconditioner inspired by the VQ rotation trick."""

    @staticmethod
    def forward(ctx, source: torch.Tensor, anchor: torch.Tensor, eps: float):
        ctx.save_for_backward(source.detach(), anchor.detach())
        ctx.eps = eps
        return source

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        source, anchor = ctx.saved_tensors
        eps = ctx.eps
        scale = (anchor.norm(dim=-1, keepdim=True) / source.norm(dim=-1, keepdim=True).clamp_min(eps)).clamp(
            0.25, 4.0
        )
        grad = _rotate_grad_between(grad_output, anchor, source, eps) * scale
        return grad, None, None


@dataclass
class ReconstructionHistory:
    initial_hard_mse: float
    final_hard_mse: float
    best_hard_mse: float
    soft_mse: list[float]
    hard_mse: list[float]

    @property
    def decreased(self) -> bool:
        return self.best_hard_mse < self.initial_hard_mse


class FactorizedIQuantGSVQ(nn.Module):
    """Gumbel-softmax VQ over IQuant-style magnitude and sign codebooks.

    This is intentionally scale-frozen. It optimizes only discrete assignments:
    magnitude-grid choices and sign-pattern choices. That makes the first
    checkpoint directly testable: reconstruction error must go down without
    changing GGUF/IQuant double-quant scales.
    """

    def __init__(
        self,
        target_vectors: torch.Tensor,
        scales: torch.Tensor,
        magnitude_codebook: torch.Tensor,
        sign_codebook: torch.Tensor,
        init_magnitude_indices: torch.Tensor,
        init_sign_indices: torch.Tensor,
        *,
        importance: Optional[torch.Tensor] = None,
        candidate_count: int = 16,
        neighbor_candidates: int = 8,
        target_candidates: int = 8,
        std: float = 0.01,
        strength: float = 0.25,
        logits_dtype: torch.dtype = torch.float32,
        rotation_trick: bool = False,
        chunk_size: int = 8192,
        output_shape: Optional[tuple[int, ...]] = None,
    ):
        super().__init__()
        if target_vectors.ndim != 2:
            raise ValueError("target_vectors must have shape (num_vectors, vector_dim)")
        if scales.ndim == 1:
            scales = scales[:, None]
        if scales.shape[0] != target_vectors.shape[0] or scales.shape[-1] not in (1, target_vectors.shape[-1]):
            raise ValueError("scales must broadcast over target_vectors")
        if magnitude_codebook.ndim != 2 or sign_codebook.ndim != 2:
            raise ValueError("codebooks must be rank-2 tensors")
        if magnitude_codebook.shape[-1] != target_vectors.shape[-1]:
            raise ValueError("magnitude codebook dimension does not match target_vectors")
        if sign_codebook.shape[-1] != target_vectors.shape[-1]:
            raise ValueError("sign codebook dimension does not match target_vectors")

        target_vectors = target_vectors.float()
        scales = scales.float()
        magnitude_codebook = magnitude_codebook.float()
        sign_codebook = sign_codebook.float()
        if importance is not None:
            if importance.ndim == 1:
                importance = importance[:, None].expand_as(target_vectors)
            importance = importance.float()
            if importance.shape != target_vectors.shape:
                raise ValueError("importance must broadcast to target_vectors")

        self.register_buffer("target_vectors", target_vectors)
        self.register_buffer("scales", scales)
        self.register_buffer("magnitude_codebook", magnitude_codebook)
        self.register_buffer("sign_codebook", sign_codebook)
        self.register_buffer("importance", importance if importance is not None else torch.ones_like(target_vectors))
        self.rotation_trick = rotation_trick
        self.output_shape = tuple(output_shape) if output_shape is not None else None
        self.eps = 1e-6

        normalized_target = target_vectors / scales.clamp_min(self.eps)
        magnitude_target = normalized_target.abs()
        sign_target = torch.where(normalized_target >= 0, torch.ones_like(normalized_target), -torch.ones_like(normalized_target))

        mag_candidates = build_candidate_indices(
            magnitude_target,
            magnitude_codebook,
            init_magnitude_indices.to(target_vectors.device),
            candidate_count=candidate_count,
            neighbor_candidates=neighbor_candidates,
            target_candidates=target_candidates,
            weights=self.importance,
            chunk_size=chunk_size,
        )
        sign_candidates = build_candidate_indices(
            sign_target,
            sign_codebook,
            init_sign_indices.to(target_vectors.device),
            candidate_count=min(candidate_count, sign_codebook.shape[0]),
            neighbor_candidates=min(neighbor_candidates, sign_codebook.shape[0]),
            target_candidates=min(target_candidates, sign_codebook.shape[0]),
            weights=self.importance,
            chunk_size=chunk_size,
        )
        self.register_buffer("magnitude_candidate_indices", mag_candidates)
        self.register_buffer("sign_candidate_indices", sign_candidates)

        mag_init_logits = self._initial_logits(mag_candidates, init_magnitude_indices, std, strength, logits_dtype)
        sign_init_logits = self._initial_logits(sign_candidates, init_sign_indices, std, strength, logits_dtype)
        self.magnitude_logits = nn.Parameter(mag_init_logits)
        self.sign_logits = nn.Parameter(sign_init_logits)

    @staticmethod
    def _initial_logits(
        candidate_indices: torch.Tensor,
        init_indices: torch.Tensor,
        std: float,
        strength: float,
        logits_dtype: torch.dtype,
    ) -> torch.Tensor:
        init_indices = init_indices.to(candidate_indices.device).reshape(-1, 1)
        logits = torch.zeros(candidate_indices.shape, device=candidate_indices.device, dtype=torch.float32)
        logits = logits.masked_fill(candidate_indices == init_indices, float(strength))
        logits = logits + std * torch.randn_like(logits)
        return logits.to(logits_dtype).detach()

    def _sample_codebook(
        self,
        logits: torch.Tensor,
        candidate_indices: torch.Tensor,
        codebook: torch.Tensor,
        temperature: float,
        logit_scale: float,
    ) -> torch.Tensor:
        eps = 1e-8
        noise = -torch.log(-torch.log(torch.rand_like(logits.float()) + eps) + eps)
        probs = F.softmax((logits.float() * float(logit_scale) + noise) / float(temperature), dim=-1)
        candidates = codebook[candidate_indices]
        return (probs[..., None] * candidates).sum(dim=1)

    def _hard_codebook(
        self,
        logits: torch.Tensor,
        candidate_indices: torch.Tensor,
        codebook: torch.Tensor,
    ) -> torch.Tensor:
        row = torch.arange(logits.shape[0], device=logits.device)
        selected = candidate_indices[row, logits.argmax(dim=-1)]
        return codebook[selected]

    def forward_vectors(self, temperature: float = 1.0, scale: float = 1.0) -> torch.Tensor:
        mag = self._sample_codebook(
            self.magnitude_logits,
            self.magnitude_candidate_indices,
            self.magnitude_codebook,
            temperature,
            scale,
        )
        sign = self._sample_codebook(
            self.sign_logits,
            self.sign_candidate_indices,
            self.sign_codebook,
            temperature,
            scale,
        )
        out = self.scales * mag * sign
        if self.rotation_trick:
            out = RotationGradient.apply(out, self.target_vectors, self.eps)
        return out

    def forward(self, temperature: float = 1.0, scale: float = 1.0) -> torch.Tensor:
        out = self.forward_vectors(temperature, scale)
        if self.output_shape is not None:
            return out.reshape(self.output_shape)
        return out

    def get_hard_vectors(self) -> torch.Tensor:
        mag = self._hard_codebook(
            self.magnitude_logits,
            self.magnitude_candidate_indices,
            self.magnitude_codebook,
        )
        sign = self._hard_codebook(
            self.sign_logits,
            self.sign_candidate_indices,
            self.sign_codebook,
        )
        return self.scales * mag * sign

    def get_hard_indices(self) -> dict[str, torch.Tensor]:
        row = torch.arange(self.magnitude_logits.shape[0], device=self.magnitude_logits.device)
        return {
            "magnitude_indices": self.magnitude_candidate_indices[row, self.magnitude_logits.argmax(dim=-1)],
            "sign_indices": self.sign_candidate_indices[row, self.sign_logits.argmax(dim=-1)],
        }

    def get_hard_weights(self):
        hard = self.get_hard_vectors()
        if self.output_shape is not None:
            hard = hard.reshape(self.output_shape)
        return hard, {
            "magnitude_candidate_indices": self.magnitude_candidate_indices,
            "sign_candidate_indices": self.sign_candidate_indices,
            "magnitude_logits": self.magnitude_logits.detach(),
            "sign_logits": self.sign_logits.detach(),
            "scales": self.scales,
        }

    def reconstruction_mse(self, hard: bool = True) -> torch.Tensor:
        out = self.get_hard_vectors() if hard else self.forward_vectors()
        return ((out - self.target_vectors).square() * self.importance).mean()


class PairedMagnitudeIQuantGSVQ(nn.Module):
    """GSVQ for IQ3-style 8D vectors made from two 4D magnitude codes.

    IQ3 GGUF blocks pair two 4D grid entries with one 8D sign pattern. This
    factorization avoids a Cartesian-product codebook and keeps the candidate
    search in the low-dimensional spaces where VQ is still tractable.
    """

    def __init__(
        self,
        target_vectors: torch.Tensor,
        scales: torch.Tensor,
        half_magnitude_codebook: torch.Tensor,
        sign_codebook: torch.Tensor,
        init_first_magnitude_indices: torch.Tensor,
        init_second_magnitude_indices: torch.Tensor,
        init_sign_indices: torch.Tensor,
        *,
        importance: Optional[torch.Tensor] = None,
        candidate_count: int = 16,
        neighbor_candidates: int = 8,
        target_candidates: int = 8,
        std: float = 0.01,
        strength: float = 0.25,
        logits_dtype: torch.dtype = torch.float32,
        rotation_trick: bool = False,
        chunk_size: int = 8192,
        output_shape: Optional[tuple[int, ...]] = None,
    ):
        super().__init__()
        if target_vectors.ndim != 2 or target_vectors.shape[-1] % 2 != 0:
            raise ValueError("target_vectors must have shape (num_vectors, even_vector_dim)")
        half_dim = target_vectors.shape[-1] // 2
        if half_magnitude_codebook.ndim != 2 or half_magnitude_codebook.shape[-1] != half_dim:
            raise ValueError("half_magnitude_codebook must match half the target vector dimension")
        if sign_codebook.ndim != 2 or sign_codebook.shape[-1] != target_vectors.shape[-1]:
            raise ValueError("sign_codebook must match target vector dimension")
        if scales.ndim == 1:
            scales = scales[:, None]

        target_vectors = target_vectors.float()
        scales = scales.float()
        half_magnitude_codebook = half_magnitude_codebook.float()
        sign_codebook = sign_codebook.float()
        if importance is not None:
            if importance.ndim == 1:
                importance = importance[:, None].expand_as(target_vectors)
            importance = importance.float()
            if importance.shape != target_vectors.shape:
                raise ValueError("importance must broadcast to target_vectors")

        self.register_buffer("target_vectors", target_vectors)
        self.register_buffer("scales", scales)
        self.register_buffer("half_magnitude_codebook", half_magnitude_codebook)
        self.register_buffer("sign_codebook", sign_codebook)
        self.register_buffer("importance", importance if importance is not None else torch.ones_like(target_vectors))
        self.rotation_trick = rotation_trick
        self.output_shape = tuple(output_shape) if output_shape is not None else None
        self.eps = 1e-6

        normalized_target = target_vectors / scales.clamp_min(self.eps)
        magnitude_target = normalized_target.abs()
        sign_target = torch.where(normalized_target >= 0, torch.ones_like(normalized_target), -torch.ones_like(normalized_target))

        first_candidates = build_candidate_indices(
            magnitude_target[:, :half_dim],
            half_magnitude_codebook,
            init_first_magnitude_indices.to(target_vectors.device),
            candidate_count=candidate_count,
            neighbor_candidates=neighbor_candidates,
            target_candidates=target_candidates,
            weights=self.importance[:, :half_dim],
            chunk_size=chunk_size,
        )
        second_candidates = build_candidate_indices(
            magnitude_target[:, half_dim:],
            half_magnitude_codebook,
            init_second_magnitude_indices.to(target_vectors.device),
            candidate_count=candidate_count,
            neighbor_candidates=neighbor_candidates,
            target_candidates=target_candidates,
            weights=self.importance[:, half_dim:],
            chunk_size=chunk_size,
        )
        sign_candidates = build_candidate_indices(
            sign_target,
            sign_codebook,
            init_sign_indices.to(target_vectors.device),
            candidate_count=min(candidate_count, sign_codebook.shape[0]),
            neighbor_candidates=min(neighbor_candidates, sign_codebook.shape[0]),
            target_candidates=min(target_candidates, sign_codebook.shape[0]),
            weights=self.importance,
            chunk_size=chunk_size,
        )
        self.register_buffer("first_magnitude_candidate_indices", first_candidates)
        self.register_buffer("second_magnitude_candidate_indices", second_candidates)
        self.register_buffer("sign_candidate_indices", sign_candidates)

        self.first_magnitude_logits = nn.Parameter(
            FactorizedIQuantGSVQ._initial_logits(
                first_candidates, init_first_magnitude_indices, std, strength, logits_dtype
            )
        )
        self.second_magnitude_logits = nn.Parameter(
            FactorizedIQuantGSVQ._initial_logits(
                second_candidates, init_second_magnitude_indices, std, strength, logits_dtype
            )
        )
        self.sign_logits = nn.Parameter(
            FactorizedIQuantGSVQ._initial_logits(sign_candidates, init_sign_indices, std, strength, logits_dtype)
        )

    def _sample_codebook(self, logits, candidate_indices, codebook, temperature, logit_scale):
        return FactorizedIQuantGSVQ._sample_codebook(self, logits, candidate_indices, codebook, temperature, logit_scale)

    def _hard_codebook(self, logits, candidate_indices, codebook):
        return FactorizedIQuantGSVQ._hard_codebook(self, logits, candidate_indices, codebook)

    def forward_vectors(self, temperature: float = 1.0, scale: float = 1.0) -> torch.Tensor:
        first = self._sample_codebook(
            self.first_magnitude_logits,
            self.first_magnitude_candidate_indices,
            self.half_magnitude_codebook,
            temperature,
            scale,
        )
        second = self._sample_codebook(
            self.second_magnitude_logits,
            self.second_magnitude_candidate_indices,
            self.half_magnitude_codebook,
            temperature,
            scale,
        )
        sign = self._sample_codebook(
            self.sign_logits,
            self.sign_candidate_indices,
            self.sign_codebook,
            temperature,
            scale,
        )
        out = self.scales * torch.cat([first, second], dim=-1) * sign
        if self.rotation_trick:
            out = RotationGradient.apply(out, self.target_vectors, self.eps)
        return out

    def forward(self, temperature: float = 1.0, scale: float = 1.0) -> torch.Tensor:
        out = self.forward_vectors(temperature, scale)
        if self.output_shape is not None:
            return out.reshape(self.output_shape)
        return out

    def get_hard_vectors(self) -> torch.Tensor:
        first = self._hard_codebook(
            self.first_magnitude_logits,
            self.first_magnitude_candidate_indices,
            self.half_magnitude_codebook,
        )
        second = self._hard_codebook(
            self.second_magnitude_logits,
            self.second_magnitude_candidate_indices,
            self.half_magnitude_codebook,
        )
        sign = self._hard_codebook(self.sign_logits, self.sign_candidate_indices, self.sign_codebook)
        return self.scales * torch.cat([first, second], dim=-1) * sign

    def get_hard_indices(self) -> dict[str, torch.Tensor]:
        row = torch.arange(self.first_magnitude_logits.shape[0], device=self.first_magnitude_logits.device)
        return {
            "first_magnitude_indices": self.first_magnitude_candidate_indices[
                row, self.first_magnitude_logits.argmax(dim=-1)
            ],
            "second_magnitude_indices": self.second_magnitude_candidate_indices[
                row, self.second_magnitude_logits.argmax(dim=-1)
            ],
            "sign_indices": self.sign_candidate_indices[row, self.sign_logits.argmax(dim=-1)],
        }

    def get_hard_weights(self):
        hard = self.get_hard_vectors()
        if self.output_shape is not None:
            hard = hard.reshape(self.output_shape)
        return hard, {
            "first_magnitude_candidate_indices": self.first_magnitude_candidate_indices,
            "second_magnitude_candidate_indices": self.second_magnitude_candidate_indices,
            "sign_candidate_indices": self.sign_candidate_indices,
            "first_magnitude_logits": self.first_magnitude_logits.detach(),
            "second_magnitude_logits": self.second_magnitude_logits.detach(),
            "sign_logits": self.sign_logits.detach(),
            "scales": self.scales,
        }

    def reconstruction_mse(self, hard: bool = True) -> torch.Tensor:
        out = self.get_hard_vectors() if hard else self.forward_vectors()
        return ((out - self.target_vectors).square() * self.importance).mean()


def train_gsvq_reconstruction(
    quantizer: FactorizedIQuantGSVQ,
    *,
    steps: int = 200,
    lr: float = 0.05,
    temperature: tuple[float, float] = (1.5, 0.1),
    logit_scale: tuple[float, float] = (1.0, 40.0),
    weight_decay: float = 0.0,
    optimizer_name: str = "adamw",
    grad_clip: Optional[float] = None,
) -> ReconstructionHistory:
    """Optimize GSVQ assignments and return hard reconstruction metrics."""
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(quantizer.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(quantizer.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(quantizer.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name == "lion":
        from lion_pytorch import Lion

        optimizer = Lion(quantizer.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer_name={optimizer_name!r}")
    initial = quantizer.reconstruction_mse(hard=True).item()
    best = initial
    best_state: Optional[Dict[str, torch.Tensor]] = None
    soft_history = []
    hard_history = []

    for step in range(int(steps)):
        t = step / max(1, int(steps) - 1)
        tau = temperature[0] + (temperature[1] - temperature[0]) * t
        scl = logit_scale[0] + (logit_scale[1] - logit_scale[0]) * t
        optimizer.zero_grad(set_to_none=True)
        out = quantizer.forward_vectors(tau, scl)
        loss = ((out - quantizer.target_vectors).square() * quantizer.importance).mean()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(quantizer.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            hard = quantizer.reconstruction_mse(hard=True).item()
        soft_history.append(float(loss.item()))
        hard_history.append(float(hard))
        if hard < best:
            best = hard
            best_state = {k: v.detach().clone() for k, v in quantizer.state_dict().items()}

    if best_state is not None:
        quantizer.load_state_dict(best_state)

    final = quantizer.reconstruction_mse(hard=True).item()
    return ReconstructionHistory(
        initial_hard_mse=float(initial),
        final_hard_mse=float(final),
        best_hard_mse=float(best),
        soft_mse=soft_history,
        hard_mse=hard_history,
    )


def build_synthetic_iquant_problem(
    *,
    qtype_name: str = "IQ2_XS",
    num_vectors: int = 4096,
    seed: int = 0,
    noise_std: float = 0.03,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Create a controlled IQuant-like reconstruction task.

    Targets are generated from true magnitude/sign codes plus small dense noise.
    Initial codes are deliberately offset, so a correct GSVQ optimizer should
    decrease hard reconstruction error without touching scales.
    """
    device = _as_device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    magnitude_codebook = load_iquant_magnitude_codebook(qtype_name, device=device)
    dim = magnitude_codebook.shape[-1]
    sign_codebook = all_sign_patterns(dim, device=device)

    true_mag = torch.randint(magnitude_codebook.shape[0], (num_vectors,), generator=generator, device=device)
    true_sign = torch.randint(sign_codebook.shape[0], (num_vectors,), generator=generator, device=device)
    scales = (0.0025 + 0.0075 * torch.rand(num_vectors, 1, generator=generator, device=device)).float()
    clean = scales * magnitude_codebook[true_mag] * sign_codebook[true_sign]
    target = clean + noise_std * scales * torch.randn(clean.shape, generator=generator, device=device)

    init_mag = (true_mag + torch.randint(1, 17, (num_vectors,), generator=generator, device=device)) % magnitude_codebook.shape[0]
    bit_to_flip = torch.randint(0, dim, (num_vectors,), generator=generator, device=device)
    init_sign = true_sign ^ (1 << bit_to_flip)
    importance = torch.ones_like(target)

    return {
        "target_vectors": target,
        "scales": scales,
        "magnitude_codebook": magnitude_codebook,
        "sign_codebook": sign_codebook,
        "init_magnitude_indices": init_mag,
        "init_sign_indices": init_sign,
        "importance": importance,
        "true_magnitude_indices": true_mag,
        "true_sign_indices": true_sign,
    }


def format_history(history: ReconstructionHistory, checkpoints: Iterable[int] = (0, 1, 5, 10, 25, 50, 100)) -> str:
    lines = [
        f"initial_hard_mse={history.initial_hard_mse:.8e}",
        f"best_hard_mse={history.best_hard_mse:.8e}",
        f"final_hard_mse={history.final_hard_mse:.8e}",
    ]
    n = len(history.hard_mse)
    for idx in checkpoints:
        if idx < n:
            lines.append(
                f"step_{idx:04d}: soft_mse={history.soft_mse[idx]:.8e} "
                f"hard_mse={history.hard_mse[idx]:.8e}"
            )
    if n:
        lines.append(
            f"step_{n - 1:04d}: soft_mse={history.soft_mse[-1]:.8e} "
            f"hard_mse={history.hard_mse[-1]:.8e}"
        )
    return "\n".join(lines)
