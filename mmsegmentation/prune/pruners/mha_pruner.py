import torch

from torch.nn.modules.batchnorm import _BatchNorm
from mmpretrain.models.utils.attention import MultiheadAttention


class MHAPruner:
    def __init__(self, model: torch.nn.Module, **kwargs) -> None:
        for n, m in model.named_modules():
            if isinstance(m, MultiheadAttention):
                self.embed_dim = m.embed_dims
                self.num_heads = m.num_heads
                self.head_dim = m.head_dims
                break

    def get_group_size(self, **kwargs):
        return 1

    def prune_index(self, idxes, p2m, group, following_group, group_size=None):
        for gi in group:
            gi = p2m[gi]
            out_mask = gi.out_mask
            out_mask.view((-1, self.num_heads, 3))[idxes] = 0.0
        for gi in following_group:
            gi = p2m[gi]
            if hasattr(gi, "in_mask"):
                gi.in_mask.view((-1, self.num_heads))[idxes] = 0.0

    def calc_group_score(self, scores, group, p2m, group_size=None):
        local_scores = []

        for gi in group:
            score = scores[gi]
            pm = p2m[gi]
            masked_s = score.clone().detach()
            masked_s[pm.out_mask.view((-1)) == 0.0] = 1000000000
            masked_s = masked_s.view((-1, self.num_heads, 3)).sum(dim=[1, 2])
            local_scores.append(masked_s)
        return torch.mean(torch.stack(local_scores), dim=0)
