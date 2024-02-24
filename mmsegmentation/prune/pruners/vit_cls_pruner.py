from typing import Any
import torch

from mmpretrain.models.backbones.vision_transformer import VisionTransformer


class ViTCLSPruner:
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
            out_mask.view((-1))[idxes] = 0.0
            if isinstance(gi, VisionTransformer):
                out_mask = out_mask.view(1, 1, -1)
                gi.cls_token.data.mul_(out_mask)
                gi.pos_embed.data.mul_(out_mask)
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
            local_scores.append(masked_s)
        return torch.mean(torch.stack(local_scores), dim=0)
