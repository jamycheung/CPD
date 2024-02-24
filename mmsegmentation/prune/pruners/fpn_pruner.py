import math
from typing import Any
import torch

from torch.nn.modules.batchnorm import _BatchNorm


class FPNPruner:
    def __init__(self, **kwargs) -> None:
        pass

    def get_group_size(self, **kwargs):
        return 16

    def prune_index(self, idxes, p2m, group, following_group, group_size=None):
        fpn_idx = -1
        for gi in group:
            name = p2m[gi].name
            fpn_idx = int(name.split(".")[2])
            gi = p2m[gi]
            out_mask = gi.out_mask
            out_mask.view((-1))[idxes] = 0.0
            out_mask = out_mask.view(-1, 1, 1, 1)
            if isinstance(gi, _BatchNorm):
                out_mask = out_mask.view(-1)
            gi.weight.data.mul_(out_mask)
            if hasattr(gi, "bias") and gi.bias is not None:
                gi.bias.data.mul_(out_mask.view((-1)))
        for gi in following_group:
            gi = p2m[gi]
            if hasattr(gi, "in_mask"):
                assert fpn_idx != -1
                adj_idxes = idxes + 512 * (1 + fpn_idx)
                gi.in_mask.view((-1))[adj_idxes] = 0.0

    def calc_group_score(self, scores, group, p2m, group_size=None) -> Any:
        local_scores = []

        for gi in group:
            score = scores[gi]
            masked_s = score.clone().detach()
            masked_s[p2m[gi].out_mask.view((-1)) == 0.0] = 1000000000
            masked_s = masked_s.view((-1))
            local_scores.append(masked_s)
        return torch.mean(torch.stack(local_scores), dim=0)
