from typing import Any

from mmseg.models.backbones.seaformer import (
    Sea_Attention,
    SqueezeAxialPositionalEmbedding,
)
from . import BasePruner
import torch
from torch.nn.modules.batchnorm import _BatchNorm


class SeaPruner(BasePruner):
    def __init__(self, model: torch.nn.Module) -> None:
        self.attn_modules: dict[str, Sea_Attention] = {}
        for n, m in model.named_modules():
            if isinstance(m, Sea_Attention):
                self.attn_modules[n] = m
                m.register_buffer("pruned_key_dim", torch.tensor(m.key_dim))

    def get_group_size(self, **kwargs):
        return 1

    def prune_index(self, idxes, p2m, group, following_group, group_size=None):
        module = self.get_parent_module(p2m[group[0]])
        for gi in group:
            gi = p2m[gi]
            if isinstance(gi, SqueezeAxialPositionalEmbedding):
                mask = gi.out_mask
                mask.view((module.num_heads, module.key_dim, -1))[:, idxes] = 0.0
                gi.pos_embed.data.mul_(mask.view((1, -1, 1)))
            else:
                mask = gi.out_mask
                mask = mask.view((module.num_heads, module.key_dim, -1))
                mask[:, idxes] = 0.0
                mask = mask.view(-1, 1, 1, 1)
                if isinstance(gi, _BatchNorm):
                    mask = mask.view(-1)
                gi.weight.data.mul_(mask)
                if hasattr(gi, "bias") and gi.bias is not None:
                    gi.bias.data.mul_(mask.view((-1)))
                if isinstance(gi, _BatchNorm):
                    gi.running_mean.data.mul_(mask.view((-1)))
                    gi.running_var.data.mul_(mask.view((-1)))
        for gi in following_group:
            gi = p2m[gi]
            if hasattr(gi, "in_mask"):
                gi.in_mask.view((module.num_heads, module.key_dim, -1))[
                    :,
                    idxes,
                ] = 0.0
        module.pruned_key_dim -= len(idxes)

    def calc_group_score(self, scores, group, p2m, group_size=None) -> Any:
        module = self.get_parent_module(p2m[group[0]])
        local_scores = []
        for gi in group:
            score = scores[gi]
            out_mask = p2m[gi].out_mask.view((-1))

            score = score.view((-1))
            masked_s = score.clone().detach()
            masked_s[out_mask == 0.0] = 1000000000
            masked_s = (
                masked_s.view((module.num_heads, module.key_dim, -1))
                .mean(dim=-1)
                .mean(dim=0)
            )
            local_scores.append(masked_s)
        return torch.mean(torch.stack(local_scores), dim=0)

    def get_parent_module(self, op):
        module = None
        for m_name, m in self.attn_modules.items():
            if m_name in op.name:
                module = m
                break
        if module is None:
            raise ValueError("Couldn't find matching parent Sea_Attention module")
        return module
