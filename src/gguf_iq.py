from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from src.quantization.gsvq import (
    FactorizedIQuantGSVQ,
    PairedMagnitudeIQuantGSVQ,
    all_sign_patterns,
    load_iquant_magnitude_codebook,
)


GGUF_TYPE_TO_NAME = {
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    29: "IQ1_M",
}


IQUANT_BITS = {
    "IQ1_S": 1.5625,
    "IQ1_M": 1.75,
    "IQ2_XXS": 2.0625,
    "IQ2_XS": 2.3125,
    "IQ2_S": 2.5,
    "IQ2_M": 2.7,
    "IQ3_XXS": 3.0625,
    "IQ3_S": 3.44,
    "IQ3_M": 3.66,
    "IQ4_XS": 4.25,
    "IQ4_NL": 4.56,
}


QK_K = 256


def gguf_name_to_hf_name(name: str) -> Optional[str]:
    if name == "token_embd.weight":
        return "model.embed_tokens"
    if name == "output.weight":
        return "lm_head"
    if name == "output_norm.weight":
        return "model.norm"

    parts = name.split(".")
    if len(parts) < 4 or parts[0] != "blk" or not parts[1].isdigit():
        return None

    layer = parts[1]
    mapping = {
        "attn_q.weight": "self_attn.q_proj",
        "attn_k.weight": "self_attn.k_proj",
        "attn_v.weight": "self_attn.v_proj",
        "attn_output.weight": "self_attn.o_proj",
        "ffn_gate.weight": "mlp.gate_proj",
        "ffn_up.weight": "mlp.up_proj",
        "ffn_down.weight": "mlp.down_proj",
        "attn_norm.weight": "input_layernorm",
        "ffn_norm.weight": "post_attention_layernorm",
    }
    suffix = mapping.get(".".join(parts[2:]))
    if suffix is None:
        return None
    return f"model.layers.{layer}.{suffix}"


def hf_name_to_gguf_name(name: str) -> Optional[str]:
    if name == "model.embed_tokens":
        return "token_embd.weight"
    if name == "lm_head":
        return "output.weight"
    if name == "model.norm":
        return "output_norm.weight"

    parts = name.split(".")
    if len(parts) < 5 or parts[0] != "model" or parts[1] != "layers":
        return None

    layer = parts[2]
    mapping = {
        "self_attn.q_proj": "attn_q.weight",
        "self_attn.k_proj": "attn_k.weight",
        "self_attn.v_proj": "attn_v.weight",
        "self_attn.o_proj": "attn_output.weight",
        "mlp.gate_proj": "ffn_gate.weight",
        "mlp.up_proj": "ffn_up.weight",
        "mlp.down_proj": "ffn_down.weight",
        "input_layernorm": "attn_norm.weight",
        "post_attention_layernorm": "ffn_norm.weight",
    }
    suffix = mapping.get(".".join(parts[3:]))
    if suffix is None:
        return None
    return f"blk.{layer}.{suffix}"


def quant_type_name(tensor_type) -> Optional[str]:
    value = getattr(tensor_type, "value", tensor_type)
    return GGUF_TYPE_TO_NAME.get(int(value))


@dataclass(frozen=True)
class GGUFIQuantTensorRef:
    gguf_name: str
    hf_name: Optional[str]
    qtype_name: str
    bits_per_weight: float
    shape: tuple[int, ...]


@dataclass
class IQuantDecomposedTensor:
    gguf_name: str
    hf_name: Optional[str]
    qtype_name: str
    dense_init: torch.Tensor
    vector_scales: torch.Tensor
    magnitude_indices: torch.Tensor
    sign_indices: torch.Tensor
    magnitude_codebook: torch.Tensor
    sign_codebook: torch.Tensor
    original_shape: tuple[int, ...]

    @property
    def vector_dim(self) -> int:
        return self.magnitude_codebook.shape[-1]

    def to_quantizer(
        self,
        target_weight: torch.Tensor,
        *,
        importance: Optional[torch.Tensor] = None,
        candidate_count: int = 16,
        neighbor_candidates: int = 8,
        target_candidates: int = 8,
        rotation_trick: bool = False,
        logits_dtype: torch.dtype = torch.float32,
    ) -> FactorizedIQuantGSVQ:
        if tuple(target_weight.shape) != self.original_shape:
            raise ValueError(
                f"target_weight shape {tuple(target_weight.shape)} does not match "
                f"GGUF tensor shape {self.original_shape}"
            )
        target_vectors = target_weight.reshape(-1, self.vector_dim).to(self.vector_scales.device)
        imp_vectors = None
        if importance is not None:
            imp_vectors = importance.reshape(-1, self.vector_dim).to(self.vector_scales.device)
        return FactorizedIQuantGSVQ(
            target_vectors,
            self.vector_scales,
            self.magnitude_codebook,
            self.sign_codebook,
            self.magnitude_indices,
            self.sign_indices,
            importance=imp_vectors,
            candidate_count=candidate_count,
            neighbor_candidates=neighbor_candidates,
            target_candidates=target_candidates,
            rotation_trick=rotation_trick,
            logits_dtype=logits_dtype,
            output_shape=self.original_shape,
        )


@dataclass
class IQuantPairedDecomposedTensor:
    gguf_name: str
    hf_name: Optional[str]
    qtype_name: str
    dense_init: torch.Tensor
    vector_scales: torch.Tensor
    first_magnitude_indices: torch.Tensor
    second_magnitude_indices: torch.Tensor
    sign_indices: torch.Tensor
    half_magnitude_codebook: torch.Tensor
    sign_codebook: torch.Tensor
    original_shape: tuple[int, ...]

    @property
    def vector_dim(self) -> int:
        return self.sign_codebook.shape[-1]

    def to_quantizer(
        self,
        target_weight: torch.Tensor,
        *,
        importance: Optional[torch.Tensor] = None,
        candidate_count: int = 16,
        neighbor_candidates: int = 8,
        target_candidates: int = 8,
        rotation_trick: bool = False,
        logits_dtype: torch.dtype = torch.float32,
    ) -> PairedMagnitudeIQuantGSVQ:
        if tuple(target_weight.shape) != self.original_shape:
            raise ValueError(
                f"target_weight shape {tuple(target_weight.shape)} does not match "
                f"GGUF tensor shape {self.original_shape}"
            )
        target_vectors = target_weight.reshape(-1, self.vector_dim).to(self.vector_scales.device)
        imp_vectors = None
        if importance is not None:
            imp_vectors = importance.reshape(-1, self.vector_dim).to(self.vector_scales.device)
        return PairedMagnitudeIQuantGSVQ(
            target_vectors,
            self.vector_scales,
            self.half_magnitude_codebook,
            self.sign_codebook,
            self.first_magnitude_indices,
            self.second_magnitude_indices,
            self.sign_indices,
            importance=imp_vectors,
            candidate_count=candidate_count,
            neighbor_candidates=neighbor_candidates,
            target_candidates=target_candidates,
            rotation_trick=rotation_trick,
            logits_dtype=logits_dtype,
            output_shape=self.original_shape,
        )


def _ksigns_codebook(
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    import gguf.quants as gguf_quants

    signs = np.frombuffer(gguf_quants.IQ2_XXS.ksigns, dtype=np.uint8)
    bits = (signs[:, None] >> np.arange(8, dtype=np.uint8).reshape(1, 8)) & np.uint8(1)
    signs = np.where(bits == 0, np.float32(1), np.float32(-1))
    return torch.from_numpy(signs.copy()).to(device=device, dtype=dtype)


class GGUFIQuantStore:
    """Reader/decomposer for GGUF IQuant tensors.

    Exact fixed-scale decomposition is implemented for the IQ2 family first,
    because those formats map cleanly to 8D magnitude vectors and sign codes.
    IQ3 formats need a second decomposition pass because their 4D grid codes
    are interleaved with 8D sign patterns.
    """

    def __init__(self, path: str | Path):
        import gguf

        self.path = str(Path(path).expanduser())
        self.reader = gguf.GGUFReader(self.path)
        self.tensors = {tensor.name: tensor for tensor in self.reader.tensors}

    def collect_refs(self, *, max_bits: float = 4.0) -> Dict[str, GGUFIQuantTensorRef]:
        refs: Dict[str, GGUFIQuantTensorRef] = {}
        for tensor in self.reader.tensors:
            qtype_name = quant_type_name(tensor.tensor_type)
            if qtype_name is None:
                continue
            bits = IQUANT_BITS.get(qtype_name, 32.0)
            if bits >= max_bits:
                continue
            hf_name = gguf_name_to_hf_name(tensor.name)
            refs[tensor.name] = GGUFIQuantTensorRef(
                gguf_name=tensor.name,
                hf_name=hf_name,
                qtype_name=qtype_name,
                bits_per_weight=bits,
                shape=tuple(int(x) for x in tensor.shape),
            )
        return refs

    def get_tensor(self, name: str):
        if name in self.tensors:
            return self.tensors[name]
        gguf_name = hf_name_to_gguf_name(name)
        if gguf_name is not None and gguf_name in self.tensors:
            return self.tensors[gguf_name]
        raise KeyError(f"No GGUF tensor named {name!r}")

    def dequantize_dense(
        self,
        name: str,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        import gguf

        tensor = self.get_tensor(name)
        dense = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
        return torch.from_numpy(dense.copy()).to(device=device, dtype=dtype)

    def decompose(
        self,
        name: str,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> IQuantDecomposedTensor | IQuantPairedDecomposedTensor:
        tensor = self.get_tensor(name)
        qtype_name = quant_type_name(tensor.tensor_type)
        if qtype_name == "IQ2_XS":
            return self._decompose_iq2_xs(tensor, device=device, dtype=dtype)
        if qtype_name == "IQ2_S":
            return self._decompose_iq2_s(tensor, device=device, dtype=dtype)
        if qtype_name == "IQ2_XXS":
            return self._decompose_iq2_xxs(tensor, device=device, dtype=dtype)
        if qtype_name == "IQ3_XXS":
            return self._decompose_iq3_xxs(tensor, device=device, dtype=dtype)
        if qtype_name == "IQ3_S":
            return self._decompose_iq3_s(tensor, device=device, dtype=dtype)
        raise NotImplementedError(
            f"Exact IQuant decomposition is not implemented yet for {qtype_name}. "
            "Current exact path supports IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, and IQ3_S."
        )

    @staticmethod
    def _blocks(data: np.ndarray, type_size: int) -> tuple[int, int, np.ndarray]:
        rows = data.view(np.uint8)
        flat_rows = rows.reshape(-1, rows.shape[-1])
        if flat_rows.shape[-1] % type_size != 0:
            raise ValueError(f"Bad byte row width {flat_rows.shape[-1]} for type size {type_size}")
        rows_count = flat_rows.shape[0]
        blocks_per_row = flat_rows.shape[-1] // type_size
        blocks = flat_rows.reshape(rows_count * blocks_per_row, type_size)
        return rows_count, blocks_per_row, blocks

    def _finish_decomposition(
        self,
        tensor,
        qtype_name: str,
        vector_scales: np.ndarray,
        magnitude_indices: np.ndarray,
        sign_indices: np.ndarray,
        *,
        sign_table: str,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> IQuantDecomposedTensor:
        dense = self.dequantize_dense(tensor.name, device=device, dtype=dtype)
        mag_codebook = load_iquant_magnitude_codebook(qtype_name, device=device, dtype=dtype)
        if sign_table == "ksigns":
            sign_codebook = _ksigns_codebook(device=device, dtype=dtype)
        elif sign_table == "all":
            sign_codebook = all_sign_patterns(mag_codebook.shape[-1], device=device).to(dtype)
        else:
            raise ValueError(f"Unknown sign table {sign_table!r}")

        hf_name = gguf_name_to_hf_name(tensor.name)
        return IQuantDecomposedTensor(
            gguf_name=tensor.name,
            hf_name=hf_name,
            qtype_name=qtype_name,
            dense_init=dense,
            vector_scales=torch.from_numpy(vector_scales.reshape(-1, 1).copy()).to(device=device, dtype=dtype),
            magnitude_indices=torch.from_numpy(magnitude_indices.reshape(-1).copy()).to(device=device, dtype=torch.long),
            sign_indices=torch.from_numpy(sign_indices.reshape(-1).copy()).to(device=device, dtype=torch.long),
            magnitude_codebook=mag_codebook,
            sign_codebook=sign_codebook,
            original_shape=tuple(dense.shape),
        )

    def _decompose_iq2_xs(
        self,
        tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> IQuantDecomposedTensor:
        rows, blocks_per_row, blocks = self._blocks(tensor.data, 74)
        d, rest = np.hsplit(blocks, [2])
        qs, scales = np.hsplit(rest, [2 * QK_K // 8])

        d = d.view(np.float16).astype(np.float32).reshape(rows, blocks_per_row, 1)
        qs = qs.view(np.uint16).reshape(rows, blocks_per_row, 32)
        scales = scales.reshape(rows, blocks_per_row, -1, 1) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 1, 2)
        scales = (scales & np.uint8(0x0F)).reshape(rows, blocks_per_row, 16)
        vector_scales = d * (np.float32(0.5) + scales.astype(np.float32)) * np.float32(0.25)
        vector_scales = np.repeat(vector_scales, 2, axis=-1)
        magnitude_indices = qs & np.uint16(511)
        sign_indices = qs >> np.uint16(9)
        return self._finish_decomposition(
            tensor,
            "IQ2_XS",
            vector_scales,
            magnitude_indices,
            sign_indices,
            sign_table="ksigns",
            device=device,
            dtype=dtype,
        )

    def _decompose_iq2_s(
        self,
        tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> IQuantDecomposedTensor:
        rows, blocks_per_row, blocks = self._blocks(tensor.data, 82)
        d, rest = np.hsplit(blocks, [2])
        qs, rest = np.hsplit(rest, [QK_K // 8])
        signs, rest = np.hsplit(rest, [QK_K // 8])
        qh, scales = np.hsplit(rest, [QK_K // 32])

        d = d.view(np.float16).astype(np.float32).reshape(rows, blocks_per_row, 1)
        scales = scales.reshape(rows, blocks_per_row, -1, 1) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 1, 2)
        scales = (scales & np.uint8(0x0F)).reshape(rows, blocks_per_row, 16)
        vector_scales = d * (np.float32(0.5) + scales.astype(np.float32)) * np.float32(0.25)
        vector_scales = np.repeat(vector_scales, 2, axis=-1)

        qh = qh.reshape(rows, blocks_per_row, -1, 1) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 1, 4)
        qh = (qh & np.uint8(0x03)).reshape(rows, blocks_per_row, 32)
        magnitude_indices = qs.astype(np.uint16).reshape(rows, blocks_per_row, 32) | (qh.astype(np.uint16) << 8)
        sign_indices = signs.reshape(rows, blocks_per_row, 32)
        return self._finish_decomposition(
            tensor,
            "IQ2_S",
            vector_scales,
            magnitude_indices,
            sign_indices,
            sign_table="all",
            device=device,
            dtype=dtype,
        )

    def _decompose_iq2_xxs(
        self,
        tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> IQuantDecomposedTensor:
        rows, blocks_per_row, blocks = self._blocks(tensor.data, 66)
        d, qs_bytes = np.hsplit(blocks, [2])
        d = d.view(np.float16).astype(np.float32).reshape(rows, blocks_per_row, 1)
        qs = qs_bytes.view(np.uint32).reshape(rows, blocks_per_row, 8, 2)

        vector_scales = d[:, :, :, None] * (
            np.float32(0.5) + (qs[..., 1] >> np.uint32(28)).astype(np.float32)
        )[:, :, :, None] * np.float32(0.25)
        vector_scales = np.repeat(vector_scales, 4, axis=-1).reshape(rows, blocks_per_row, 32)

        magnitude_indices = qs[..., 0].copy().view(np.uint8).reshape(rows, blocks_per_row, 8, 4)
        shifts = np.array([0, 7, 14, 21], dtype=np.uint32).reshape(1, 1, 1, 4)
        sign_indices = ((qs[..., 1][..., None] >> shifts) & np.uint32(0x7F)).astype(np.uint8)
        return self._finish_decomposition(
            tensor,
            "IQ2_XXS",
            vector_scales,
            magnitude_indices,
            sign_indices,
            sign_table="ksigns",
            device=device,
            dtype=dtype,
        )

    def _finish_paired_decomposition(
        self,
        tensor,
        qtype_name: str,
        vector_scales: np.ndarray,
        first_magnitude_indices: np.ndarray,
        second_magnitude_indices: np.ndarray,
        sign_indices: np.ndarray,
        *,
        sign_table: str,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> IQuantPairedDecomposedTensor:
        dense = self.dequantize_dense(tensor.name, device=device, dtype=dtype)
        mag_codebook = load_iquant_magnitude_codebook(qtype_name, device=device, dtype=dtype)
        if sign_table == "ksigns":
            sign_codebook = _ksigns_codebook(device=device, dtype=dtype)
        elif sign_table == "all":
            sign_codebook = all_sign_patterns(8, device=device).to(dtype)
        else:
            raise ValueError(f"Unknown sign table {sign_table!r}")

        return IQuantPairedDecomposedTensor(
            gguf_name=tensor.name,
            hf_name=gguf_name_to_hf_name(tensor.name),
            qtype_name=qtype_name,
            dense_init=dense,
            vector_scales=torch.from_numpy(vector_scales.reshape(-1, 1).copy()).to(device=device, dtype=dtype),
            first_magnitude_indices=torch.from_numpy(first_magnitude_indices.reshape(-1).copy()).to(
                device=device, dtype=torch.long
            ),
            second_magnitude_indices=torch.from_numpy(second_magnitude_indices.reshape(-1).copy()).to(
                device=device, dtype=torch.long
            ),
            sign_indices=torch.from_numpy(sign_indices.reshape(-1).copy()).to(device=device, dtype=torch.long),
            half_magnitude_codebook=mag_codebook,
            sign_codebook=sign_codebook,
            original_shape=tuple(dense.shape),
        )

    @staticmethod
    def _pair_iq3_grid_indices(qs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pairs = qs.reshape(qs.shape[0], qs.shape[1], 8, 4, 2)
        return pairs[..., 0], pairs[..., 1]

    def _decompose_iq3_xxs(
        self,
        tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> IQuantPairedDecomposedTensor:
        rows, blocks_per_row, blocks = self._blocks(tensor.data, 98)
        d, rest = np.hsplit(blocks, [2])
        qs, scales = np.hsplit(rest, [QK_K // 4])

        d = d.view(np.float16).astype(np.float32).reshape(rows, blocks_per_row, 1)
        scales = scales.view(np.uint32).reshape(rows, blocks_per_row, 8)
        vector_scales = d * (np.float32(0.5) + (scales >> np.uint32(28)).astype(np.float32)) * np.float32(0.5)
        vector_scales = np.repeat(vector_scales[:, :, :, None], 4, axis=-1)

        signs = scales[:, :, :, None] >> np.array([0, 7, 14, 21], dtype=np.uint32).reshape(1, 1, 1, 4)
        sign_indices = (signs & np.uint32(0x7F)).astype(np.uint8)
        first, second = self._pair_iq3_grid_indices(qs.reshape(rows, blocks_per_row, 64).astype(np.uint16))
        return self._finish_paired_decomposition(
            tensor,
            "IQ3_XXS",
            vector_scales,
            first,
            second,
            sign_indices,
            sign_table="ksigns",
            device=device,
            dtype=dtype,
        )

    def _decompose_iq3_s(
        self,
        tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> IQuantPairedDecomposedTensor:
        rows, blocks_per_row, blocks = self._blocks(tensor.data, 110)
        d, rest = np.hsplit(blocks, [2])
        qs, rest = np.hsplit(rest, [QK_K // 4])
        qh, rest = np.hsplit(rest, [QK_K // 32])
        signs, scales = np.hsplit(rest, [QK_K // 8])

        d = d.view(np.float16).astype(np.float32).reshape(rows, blocks_per_row, 1)
        scales = scales.reshape(rows, blocks_per_row, -1, 1) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 1, 2)
        scales = (scales & np.uint8(0x0F)).reshape(rows, blocks_per_row, 8)
        vector_scales = d * (1 + 2 * scales.astype(np.float32))
        vector_scales = np.repeat(vector_scales[:, :, :, None], 4, axis=-1)

        qh = qh.reshape(rows, blocks_per_row, -1, 1) >> np.array([i for i in range(8)], dtype=np.uint8)
        qh = (qh & np.uint8(0x01)).astype(np.uint16).reshape(rows, blocks_per_row, 64)
        qs = qs.astype(np.uint16).reshape(rows, blocks_per_row, 64) | (qh << 8)
        first, second = self._pair_iq3_grid_indices(qs)
        sign_indices = signs.reshape(rows, blocks_per_row, 8, 4)
        return self._finish_paired_decomposition(
            tensor,
            "IQ3_S",
            vector_scales,
            first,
            second,
            sign_indices,
            sign_table="all",
            device=device,
            dtype=dtype,
        )
