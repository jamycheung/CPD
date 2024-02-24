import math
from typing import Any
from mmseg.models.decode_heads.segmenter_mask_head import SegmenterMaskTransformerHead
import torch

from mmseg.models.backbones.vit import VisionTransformer
from torch.nn.modules.batchnorm import _BatchNorm


class ViTPruner:
    def __init__(self, model: torch.nn.Module, **kwargs) -> None:
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.MultiheadAttention):
                self.group_size = m.num_heads
                break

    def get_group_size(self, **kwargs):
        return self.group_size

    def prune_index(self, idxes, p2m, group, following_group, group_size=None):
        for gi in group:
            gi = p2m[gi]
            out_mask = gi.out_mask
            if isinstance(gi, torch.nn.MultiheadAttention):
                out_mask.view(3, -1)[:, idxes] = 0.0
                out_mask = out_mask.view(-1, 1)
                gi.in_proj_weight.data.mul_(out_mask)
                gi.in_proj_bias.data.mul_(out_mask.view(-1))
                continue
            out_mask.view((-1))[idxes] = 0.0
            if isinstance(gi, VisionTransformer):
                out_mask = out_mask.view(1, 1, -1)
                gi.cls_token.data.mul_(out_mask)
                gi.pos_embed.data.mul_(out_mask)
                continue
            if isinstance(gi, SegmenterMaskTransformerHead):
                out_mask = out_mask.view(1, 1, -1)
                gi.cls_emb.data.mul_(out_mask)
                continue
            out_mask = out_mask.view(-1, 1, 1, 1)
            if isinstance(gi, torch.nn.LayerNorm):
                out_mask = out_mask.view(-1)
            if isinstance(gi, torch.nn.Linear):
                out_mask = out_mask.view(-1, 1)
            gi.weight.data.mul_(out_mask)
            if hasattr(gi, "bias") and gi.bias is not None:
                gi.bias.data.mul_(out_mask.view((-1)))
        for gi in following_group:
            gi = p2m[gi]
            if hasattr(gi, "in_mask"):
                gi.in_mask.view((-1))[idxes] = 0.0

    def calc_group_score(self, scores, group, p2m, group_size=None) -> Any:
        local_scores = []

        for gi in group:
            score = scores[gi]
            pm = p2m[gi]
            masked_s = score.clone().detach()
            masked_s[pm.out_mask.view((-1)) == 0.0] = 1000000000
            masked_s = masked_s.view((-1))
            if isinstance(pm, torch.nn.MultiheadAttention):
                masked_s = masked_s.view(3, -1).sum(dim=0)
            local_scores.append(masked_s)
        return torch.mean(torch.stack(local_scores), dim=0)


class ViTNeckPruner:
    def __init__(self, **kwargs) -> None:
        pass

    def get_group_size(self, **kwargs):
        return 16

    def prune_index(self, idxes, p2m, group, following_group, group_size=None):
        psp_idx = -1
        psp_mod = p2m[group[0]]
        if "psp_modules" in psp_mod.name:
            # Get index number from name
            psp_idx = int(psp_mod.name.split(".")[2])
        for gi in group:
            gi = p2m[gi]
            out_mask = gi.out_mask
            out_mask.view((-1))[idxes] = 0.0
            out_mask = out_mask.view(-1, 1, 1, 1)
            if isinstance(gi, _BatchNorm):
                out_mask = out_mask.view(-1)
            gi.weight.data.mul_(out_mask)
            if hasattr(gi, "bias") and gi.bias is not None:
                gi.bias.data.mul_(out_mask.view((-1)))
            if isinstance(gi, _BatchNorm):
                gi.running_mean.data.mul_(out_mask.view((-1)))
                gi.running_var.data.mul_(out_mask.view((-1)))
        for gi in following_group:
            gi = p2m[gi]
            if "bottleneck" in gi.name:
                # Bottleneck conv
                if psp_idx != -1:
                    adjusted_idx = 768 + 512 * psp_idx + idxes
                else:
                    adjusted_idx = idxes
                gi.in_mask.view((-1))[adjusted_idx] = 0.0

    def calc_group_score(self, scores, group, p2m, group_size=None) -> Any:
        local_scores = []

        for gi in group:
            score = scores[gi]
            masked_s = score.clone().detach()
            masked_s[p2m[gi].out_mask.view((-1)) == 0.0] = 1000000000
            masked_s = masked_s.view((-1))
            local_scores.append(masked_s)
        return torch.mean(torch.stack(local_scores), dim=0)
