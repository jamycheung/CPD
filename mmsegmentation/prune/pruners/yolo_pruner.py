from typing import Any
import re

from . import BasePruner
import torch
from torch.nn.modules.batchnorm import _BatchNorm


class CSPTwoConvPruner(BasePruner):
    def __init__(self, model: torch.nn.Module) -> None:
        pass

    def get_group_size(self, **kwargs):
        return 1

    def prune_index(self, idxes, group, following_group, group_size=1):
        main_size = 0
        if len(group) > 1:
            for gi in group:
                if "main_conv" in gi.name:
                    main_size = gi.out_mask.numel()

        for gi in group:
            out_mask = gi.out_mask
            if main_size != 0:
                for idx in idxes:
                    if idx < main_size // 2:
                        # channel is in first half of splits, need to only prune in main_conv
                        if "main_conv" in gi.name:
                            out_mask.view((-1))[idx] = 0.0
                    else:
                        # channel is in second half of splits, need to use adjusted index for smaller (blocks) convs
                        adjusted_idx = (
                            idx if "main_conv" in gi.name else idx - main_size // 2
                        )
                        out_mask.view((-1))[adjusted_idx] = 0.0
            else:
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
            if hasattr(gi, "in_mask"):
                for idx in idxes:
                    if idx < main_size // 2:
                        # channel is in first half of splits, can just prune normally
                        gi.in_mask.view((-1))[idx] = 0.0
                    else:
                        # channel is in second half of splits, need to use adjusted index for smaller (blocks) convs
                        adjusted_idx = idx - main_size // 2
                        if main_size != 0:
                            gi.in_mask.view((main_size // 2, -1))[
                                adjusted_idx, 1:
                            ] = 0.0
                        else:
                            gi.in_mask.view(-1)[adjusted_idx] = 0.0

    def calc_group_score(self, scores, group, group_size=1) -> Any:
        local_scores = []
        score_group = group
        bns = [gi for gi in group if gi.name.endswith(".bn")]
        other_ops = [gi for gi in group if not gi.name.endswith(".bn")]
        if len(bns) == len(other_ops):
            score_group = bns
        else:
            score_group = other_ops

        main_size = 0
        if len(score_group) > 1:
            for gi in score_group:
                if "main_conv" in gi.name:
                    main_size = scores[gi].numel()

        for gi in score_group:
            score = scores[gi]
            masked_s = score.clone().detach()
            masked_s[gi.out_mask.view((-1)) == 0.0] = 1000000000
            masked_s = masked_s.view((-1))
            if masked_s.numel() < main_size:
                masked_s = torch.nn.functional.pad(masked_s, (0, main_size // 2))
            local_scores.append(masked_s)
        return torch.mean(torch.stack(local_scores), dim=0)
