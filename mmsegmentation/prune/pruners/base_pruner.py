import math
from typing import Any
import torch

from torch.nn.modules.batchnorm import _BatchNorm


def round_pow_2(x):
    if x <= 0:
        return 1
    return 2 ** math.floor(math.log(x, 2))


START_FACTOR = 2


class BasePruner:
    def __init__(self, **kwargs) -> None:
        pass

    def get_group_size(self, group, following_group, p2m, n_iter, max_iters):
        g_mask = p2m[group[0]].out_mask.view((-1))
        g_mask_els = g_mask.numel()
        g_mask_zeros = g_mask_els - torch.count_nonzero(g_mask).item()
        out_channels = -1
        conv_group_sizes = []
        for op in group:
            op = p2m[op]
            if isinstance(op, torch.nn.modules.conv._ConvNd):
                if op.groups == op.out_channels and op.out_channels > 1:
                    out_channels = op.out_channels  # DWConv
                elif op.groups > 1:
                    conv_group_sizes.append(op.groups)
                else:
                    out_channels = op.out_channels
        for op in following_group:
            op = p2m[op]
            if isinstance(op, torch.nn.modules.conv._ConvNd):
                if (
                    op.groups > 1
                    and op.groups != op.out_channels
                    and op.groups != op.in_channels
                ):
                    conv_group_sizes.append(op.groups)
        if len(conv_group_sizes):
            if len(set(conv_group_sizes)) != 1:
                raise ValueError(
                    f"Different conv group sizes in one pruning group {conv_group_sizes}"
                )
            else:
                self.conv_group_size = conv_group_sizes[0]
                return conv_group_sizes[0]
        if out_channels != -1:
            return min(round_pow_2((out_channels - g_mask_zeros) // 16), 4)
        else:
            return 1

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
            # masked_s = masked_s.view((-1)).sum(dim=1)
            local_scores.append(masked_s)
        try:
            return torch.mean(torch.stack(local_scores), dim=0)
        except:
            print([type(a) for a in group])
            print("cant get mean in basepruner")
            print(len(group))
            quit()
