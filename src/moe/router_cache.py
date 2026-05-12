from dataclasses import dataclass, field
import torch
from typing import Dict, Tuple, Optional

@dataclass
class BatchRouting:
    expert_to_tokens: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    token_positions: Optional[torch.Tensor] = None

    @staticmethod
    def from_topk(topk_ids_cpu: torch.Tensor,
                  topk_w_cpu: torch.Tensor,
                  index_dtype=torch.int32,
                  weight_dtype=torch.float16) -> "BatchRouting":
        assert topk_ids_cpu.ndim == 2 and topk_w_cpu.ndim == 2
        BT, top_k = topk_ids_cpu.shape
        e2t = {}

        for k in range(top_k):
            e_ids = topk_ids_cpu[:, k].to(index_dtype, copy=False)
            w = topk_w_cpu[:, k].to(weight_dtype, copy=False)
            uniq = torch.unique(e_ids, sorted=True)
            for eid in uniq.tolist():
                mask = (e_ids == eid)
                idxs = torch.nonzero(mask, as_tuple=False).squeeze(1).to(index_dtype)
                if idxs.numel() == 0:
                    continue
                ws = w.index_select(0, idxs)
                if eid not in e2t:
                    e2t[eid] = (idxs.contiguous(), ws.contiguous())
                else:
                    prev_i, prev_w = e2t[eid]
                    e2t[eid] = (
                        torch.cat([prev_i, idxs]).contiguous(),
                        torch.cat([prev_w, ws]).contiguous()
                    )

        return BatchRouting(
            expert_to_tokens=e2t,
            token_positions=torch.arange(BT, dtype=index_dtype)
        )


class RouterCache:
    def __init__(self, layer_idx, top_k, weight_dtype=torch.float16, index_dtype=torch.int32):
        self.layer_idx = layer_idx
        self.top_k = top_k
        self.weight_dtype = weight_dtype
        self.index_dtype = index_dtype
        self.splits = {"train": {}, "val": {}}

    def add_batch_from_topk(self, split, batch_idx, topk_ids_cpu, topk_w_cpu):
        assert split in self.splits
        br = BatchRouting.from_topk(
            topk_ids_cpu=topk_ids_cpu,
            topk_w_cpu=topk_w_cpu,
            index_dtype=self.index_dtype,
            weight_dtype=self.weight_dtype
        )
        self.splits[split][batch_idx] = br

    def get_batch(self, split, batch_idx):
        return self.splits[split][batch_idx]
