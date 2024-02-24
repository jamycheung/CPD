from collections import defaultdict
from typing import Dict
from mmseg.models.decode_heads.segmenter_mask_head import SegmenterMaskTransformerHead

import numpy as np
import prune.pruners
from prune.prune_cfg import PruneCfg
import torch
import torch.nn as nn

from mmseg.registry import HOOKS
from mmengine.runner import load_checkpoint, Runner
from mmengine.hooks import Hook
from torch.nn import Conv2d
from torch.nn.modules import Linear
from torch.nn.modules.batchnorm import _NormBase
from torch.nn.modules.normalization import LayerNorm
from mmseg.models.backbones.seaformer import (
    SqueezeAxialPositionalEmbedding,
)
from mmseg.models.backbones.vit import VisionTransformer
from mmpretrain.models.backbones.vision_transformer import VisionTransformer as CLSViT


@HOOKS.register_module()
class PruningHook(Hook):
    """Use fisher information to pruning the model, must register after
    optimizer hook.

    Args:
        pruning (bool): When True, the model in pruning process,
            when False, the model is in finetune process.
            Default: True
        delta (str): "acts" or "flops", prune the model by
            "acts" or flops. Default: "acts"
        batch_size (int): The batch_size when pruning model.
            Default: 2
        interval (int): The interval of  pruning two channels.
            Default: 10
        deploy_from (str): Path of checkpoint containing the structure
            of pruning model. Defaults to None and only effective
            when pruning is set True.
        save_flops_thr  (list): Checkpoint would be saved when
            the flops reached specific value in the list:
            Default: [0.75, 0.5, 0.25]
        save_acts_thr (list): Checkpoint would be saved when
            the acts reached specific value in the list:
            Default: [0.75, 0.5, 0.25]
    """

    def __init__(
        self,
        prune_cfg,
        pruning=True,
        delta="acts",
        interval=10,
        deploy_from=None,
        save_sparsity_thr=[0.25, 0.5, 0.75],
        continue_finetune=False,
        acts_chan_dim=1,
        metric="mIoU",
    ):
        assert delta in ("acts", "flops")
        self.pruning = pruning
        self.delta = delta
        self.interval = interval
        self.scores = defaultdict(float)
        # The key of self.acts is conv module, and value of it
        # is number of all the out feature's activations(N*C*H*W)
        # in forward process
        self.acts = defaultdict(float)
        # Number of channel dimension that will be used to calculate acts
        self.acts_chan_dim = acts_chan_dim
        # The key of self.temp_fisher_info is conv module, and value
        # is a temporary variable used to estimate fisher.
        self.temp_fisher_info = {}

        # The key of self.batch_fishers is conv module, and value
        # is the estimation of fisher by single batch.
        self.batch_fishers = {}

        # The key of self.accum_fishers is conv module, and value
        # is the estimation of parameter's fisher by all the batch
        # during number of self.interval iterations.
        self.accum_fishers = {}
        self.channels = 0
        self.delta = delta
        self.deploy_from = deploy_from

        for i in range(len(save_sparsity_thr) - 1):
            assert save_sparsity_thr[i] < save_sparsity_thr[i + 1]

        self.save_sparsity_thr = save_sparsity_thr

        self.prune_cfg = PruneCfg(file=prune_cfg)
        self.pruning_functions = {}

        self.continue_finetune = continue_finetune
        self.stop_pruning = False
        self.finetune_best_score = 0.0
        self.metric = metric

    def before_run(self, runner: Runner):
        """Initialize the relevant variables(fisher, flops and acts) for
        calculating the importance of the channel, and use the layer-grouping
        algorithm to make the coupled module shared the mask of input
        channel."""

        model = runner.model
        if hasattr(runner.model, "student"):
            model = runner.model.student
        elif hasattr(runner.model, "module"):
            model = runner.model.module
            if hasattr(runner.model.module, "architecture"):
                model = runner.model.module.architecture
        self.model = model

        self.params: dict[str, torch.nn.Parameter] = {}
        self.param2parent: dict[torch.nn.Parameter, torch.nn.Module] = {}
        self.param2name: dict[torch.nn.Parameter, str] = {}
        for mn, m in model.named_modules():
            m.name = mn
            for pn, p in m.named_parameters(prefix=mn, recurse=False):
                self.param2name[p] = pn
                self.param2parent[p] = m
                self.params[pn] = p

        self.groups = []
        for group in self.prune_cfg.groups:
            og, oig = group["out_group"], group["out_in_group"]
            param_og, param_oig = [], []
            param_og.extend([self.params[pn] for pn in og])
            param_oig.extend([self.params[pn] for pn in oig])
            self.groups.append(
                {"og": param_og, "oig": param_oig, "type": group["type"]}
            )

        self.add_masks()
        if self.pruning:
            self.register_hooks(runner)
            self.register_pruning_functions(model)

    def add_masks(self):
        for p, m in self.param2parent.items():
            if isinstance(m, Conv2d):
                m.register_buffer(
                    "out_mask",
                    m.weight.new_ones((m.out_channels, 1, 1, 1)),
                )
                m.register_buffer(
                    "in_mask", m.weight.new_ones((1, m.in_channels, 1, 1))
                )
            elif isinstance(m, Linear):
                m.register_buffer(
                    "out_mask",
                    m.weight.new_ones((m.out_features, 1)),
                )
                m.register_buffer("in_mask", m.weight.new_ones((1, m.in_features)))
            elif isinstance(m, SqueezeAxialPositionalEmbedding):
                m.register_buffer(
                    "out_mask", m.pos_embed.new_ones((1, m.pos_embed.size(1), 1))
                )
            elif isinstance(m, _NormBase) or isinstance(m, LayerNorm):
                m.register_buffer(
                    "out_mask",
                    m.weight.new_ones(m.weight.shape),
                )
            elif (
                isinstance(m, VisionTransformer)
                or isinstance(m, CLSViT)
                or isinstance(m, SegmenterMaskTransformerHead)
            ):
                if not hasattr(m, "out_mask"):
                    m.register_buffer("out_mask", p.new_ones(1, 1, p.size(-1)))
            elif isinstance(m, torch.nn.MultiheadAttention):
                m.register_buffer("out_mask", p.new_ones(3 * m.embed_dim, 1))
                m.register_buffer("in_mask", p.new_ones(1, m.embed_dim))
            else:
                raise TypeError(f"Unknown weight type: {self.param2name[p]},{type(m)}")

    def calc_group_scores(self):
        group_scores = {}
        for g in self.groups:
            if g["type"] not in self.pruning_functions:
                raise TypeError(f"Invalid pruning type {g['type']}.")
            pruner = self.pruning_functions[g["type"]]
            scores = pruner.calc_group_score(self.scores, g["og"], self.param2parent)
            group_scores[id(g)] = scores / (len(g["og"]) + len(g["oig"]))
        return group_scores

    def find_least_important_neuron(self, group_scores, runner):
        cur_min = 1000000000
        cur_idx = None
        cur_grp = None
        for g in self.groups:
            if id(g) not in group_scores:
                continue
            scores = group_scores[id(g)]
            group_size = self.pruning_functions[g["type"]].get_group_size(
                group=g["og"],
                following_group=g["oig"],
                p2m=self.param2parent,
                n_iter=runner.iter,
                max_iters=runner.max_iters,
            )

            mv, mi = torch.topk(scores, group_size, largest=False)

            aggregated_min = torch.sum(mv)
            if aggregated_min < cur_min:
                cur_min = aggregated_min
                cur_grp = g
                cur_idx = mi
        return cur_grp, cur_idx, cur_min

    def remove_indices(self, group, idx):
        pruner = self.pruning_functions[group["type"]]
        pruner.prune_index(idx, self.param2parent, group["og"], group["oig"])

    def prune(self, runner):
        if self.stop_pruning:
            return
        _, _, sparsity = self.calc_overall_sparsity()
        if sparsity >= self.save_sparsity_thr[-1]:
            return
        group_scores = self.calc_group_scores()
        group, idx, val = self.find_least_important_neuron(group_scores, runner)
        self.last_pruned = id(group)
        self.remove_indices(group, idx)

        g_mask = self.param2parent[group["og"][0]].out_mask.view((-1))
        g_mask_els = g_mask.numel()
        g_mask_zeros = g_mask_els - torch.count_nonzero(g_mask).item()
        runner.logger.info(
            f"Pruning: {group['type']} Importance: {val} Sparsity: {g_mask_zeros/g_mask_els:.4%} ({g_mask_zeros}/{g_mask_els}) Modules: {set([self.param2parent[gi].name for gi in group['og']])}",
        )

        # Reset scores
        for m in self.scores.keys():
            self.scores[m] = 0

    def after_train_iter(self, runner: Runner, **kwargs):
        if not self.pruning:
            return
        if self.every_n_train_iters(runner, self.interval):
            self.prune(runner)

    def before_val(self, runner):
        for module in self.param2parent.values():
            self.mask_out_hook(module)

    def after_val(self, runner) -> None:
        self.print_model(runner)

    def after_val_epoch(self, runner: Runner, metrics: Dict[str, float]) -> None:
        if self.stop_pruning and self.continue_finetune and metrics is not None:
            self.save_best_checkpoint(runner, metrics)

    def save_best_checkpoint(self, runner: Runner, metrics: Dict[str, float]):
        if metrics[self.metric] > self.finetune_best_score:
            self.finetune_best_score = metrics[self.metric]
            runner.save_checkpoint(runner.work_dir, "finetune_best.pth")
            runner.logger.info(
                f"Saved best model during finetune at {self.metric}: {metrics[self.metric]}",
            )

    def after_run(self, runner) -> None:
        self.print_model(runner, True)

    def print_model(self, runner: Runner, print_modules=False):
        if not self.pruning:
            return

        total_zeros = 0
        total_els = 0
        counted = set()
        for p, m in self.param2parent.items():
            mask = m.out_mask.view((-1))
            els = mask.numel()
            zeros = els - torch.count_nonzero(mask).item()
            if m.name not in counted:
                total_zeros += zeros
                total_els += els
                counted.add(m.name)
            if print_modules:
                runner.logger.info(
                    f"Op: {self.param2name[p]} - Sparsity: {zeros/els:.4%} ({zeros}/{els})",
                )
        runner.logger.info(
            f"Model overall: Sparsity: {total_zeros/total_els:.4%} ({total_zeros}/{total_els})",
        )

        if len(self.save_sparsity_thr):
            thr = self.save_sparsity_thr[0]
            if (total_zeros / total_els) >= thr:
                self.save_sparsity_thr.pop(0)
                runner.save_checkpoint(runner.work_dir, f"thr_{thr}.pth")
                if len(self.save_sparsity_thr) == 0:
                    if self.continue_finetune:
                        self.stop_pruning = True
                    else:
                        exit()
        else:
            if self.continue_finetune:
                self.stop_pruning = True
            else:
                exit()

    def calc_overall_sparsity(self):
        total_zeros = 0
        total_els = 0
        counted = set()
        for m in self.param2parent.values():
            if m.name in counted:
                continue
            mask = m.out_mask.view((-1))
            els = mask.numel()
            zeros = els - torch.count_nonzero(mask).item()
            total_zeros += zeros
            total_els += els
            counted.add(m.name)
        return total_zeros, total_els, total_zeros / total_els

    def register_hooks(self, runner: Runner):
        registered = set()
        for module in self.param2parent.values():
            if module.name in registered:
                continue
            registered.add(module.name)
            module.register_forward_pre_hook(self.mask_out_hook)
            # module.register_full_backward_hook(self.mask_backward_hook)
            module.register_forward_hook(self.save_input_forward_hook)

        orig_func = getattr(runner.optim_wrapper, "backward")

        def wrapped_func(*args, **kwargs):
            orig_func(*args, **kwargs)
            for p, module in self.param2parent.items():
                self.mask_out_hook(module)
                pd = p.detach()
                gpd = p.grad.detach()
                prod = pd * gpd
                # if self.param2name[p].endswith("bias"):
                #     self.scores[p] += prod.view(-1).square() / self.acts[module]
                if isinstance(module, Conv2d) or isinstance(module, Linear):
                    c = pd.size(0)
                    self.scores[p] += (
                        prod.view(c, -1).sum(dim=1).square() / self.acts[module]
                    )
                elif isinstance(module, SqueezeAxialPositionalEmbedding):
                    c = pd.size(1)
                    self.scores[p] += (
                        prod.view(c, -1).sum(dim=1).square() / self.acts[module]
                    )
                elif isinstance(module, _NormBase) or isinstance(module, LayerNorm):
                    c = pd.size(0)
                    self.scores[p] += (
                        prod.view(c, -1).sum(dim=1).square() / self.acts[module]
                    )
                elif isinstance(module, VisionTransformer) or isinstance(
                    module, SegmenterMaskTransformerHead
                ):
                    c = pd.size(2)
                    self.scores[p] += (
                        prod.view(1, -1, c).sum(dim=[0, 1]).square() / self.acts[module]
                    )
                elif isinstance(module, torch.nn.MultiheadAttention):
                    c = pd.size(0)
                    self.scores[p] += (
                        prod.view(c, -1).sum(dim=1).square() / self.acts[module]
                    )
                else:
                    raise TypeError("Unknown weight type")

        setattr(runner.optim_wrapper, "backward", wrapped_func)

    def save_input_forward_hook(self, module, inputs, outputs):
        c = module.out_mask.sum().item()
        if type(outputs) is tuple:
            outputs = outputs[0]
        try:
            self.acts[module] = np.prod(
                [
                    *(outputs.shape[: self.acts_chan_dim]),
                    *(outputs.shape[self.acts_chan_dim + 1 :]),
                    c,
                ]
            )
        except:
            print(type(module), type(outputs))
        if isinstance(module, torch.nn.MultiheadAttention):
            op = module.out_proj
            c = op.out_mask.sum().item()
            self.acts[op] = np.prod(
                [
                    *(outputs.shape[: self.acts_chan_dim]),
                    *(outputs.shape[self.acts_chan_dim + 1 :]),
                    c,
                ]
            )

    def mask_out_hook(self, module, *args):
        out_mask = module.out_mask
        out_mask = out_mask.view(-1, 1, 1, 1)
        if isinstance(module, _NormBase) or isinstance(module, LayerNorm):
            out_mask = out_mask.view(-1)
        if isinstance(module, SqueezeAxialPositionalEmbedding):
            module.pos_embed.data.mul_(out_mask.view((1, -1, 1)))
        elif isinstance(module, VisionTransformer):
            module.cls_token.data.mul_(out_mask.view(1, 1, -1))
            module.pos_embed.data.mul_(out_mask.view(1, 1, -1))
        elif isinstance(module, SegmenterMaskTransformerHead):
            module.cls_emb.data.mul_(out_mask.view(1, 1, -1))
        elif isinstance(module, torch.nn.MultiheadAttention):
            module.in_proj_weight.data.mul_(out_mask.view(-1, 1))
            module.in_proj_bias.data.mul_(out_mask.view(-1))
        elif isinstance(module, Linear):
            module.weight.data.mul_(out_mask.view(-1, 1))
        else:
            module.weight.data.mul_(out_mask)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.mul_(out_mask.view((-1)))
        if isinstance(module, _NormBase):
            if module.track_running_stats:
                module.running_mean.data.mul_(out_mask.view((-1)))
                module.running_var.data.mul_(out_mask.view((-1)))

    def register_pruning_functions(self, model):
        self.pruning_functions = {
            pname: getattr(prune.pruners, pname)(model=model)
            for pname in prune.pruners.__all__
        }
