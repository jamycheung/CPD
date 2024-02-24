from typing import Any
import torch

from . import BasePruner
from torch.nn.modules.batchnorm import _BatchNorm


class BiSev2SemFusePruner(BasePruner):
    def __init__(self, **kwargs) -> None:
        pass

    def prune_index(self, idxes, p2m, group, following_group, group_size=None):
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
            if hasattr(gi, "in_mask"):
                gi.in_mask.view((-1))[idxes] = 0.0
                gi.in_mask.view((-1))[idxes * 2] = 0.0

    def calc_group_score(self, scores, group, p2m, group_size=None) -> Any:
        local_scores = []
        len_min = min([scores[gi].size(0) for gi in group])

        for gi in group:
            score = scores[gi]
            masked_s = score.clone().detach()
            try:
                masked_s[p2m[gi].out_mask.view((-1)) == 0.0] = 1000000000
            except:
                print(type(p2m[gi]), p2m[gi].out_mask.shape, masked_s.shape)
            masked_s = masked_s.view((len_min, -1)).mean(dim=1)
            local_scores.append(masked_s)
        try:
            return torch.mean(torch.stack(local_scores), dim=0)
        except:
            print([type(a) for a in group])
            print("cant get mean in bisenetv2_pruner")
            print(len(group))
            quit()
