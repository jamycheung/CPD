import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn.modules.batchnorm import _BatchNorm
from mmseg.models.backbones.seaformer import (
    SqueezeAxialPositionalEmbedding,
    Sea_Attention,
)
from torch.nn.modules import Linear
from torch.nn.modules.batchnorm import _NormBase
from torch.nn.modules.normalization import LayerNorm
from mmseg.models.backbones.vit import VisionTransformer


def deploy_pruning(model):
    param2parent: dict[torch.nn.Parameter, torch.nn.Module] = {}
    for mn, m in model.named_modules():
        m.name = mn
        for _, p in m.named_parameters(prefix=mn, recurse=False):
            param2parent[p] = m

    done = []

    for module in param2parent.values():
        if "auxiliary_head" in module.name:
            # Is not used in deployment
            continue
        if isinstance(module, Sea_Attention):
            module.key_dim = module.pruned_key_dim.item()
        if module in done:
            # Prevent duplicates
            continue
        if hasattr(module, "out_mask"):
            out_mask = module.out_mask.view(-1).bool()
            if isinstance(module, VisionTransformer):
                module.cls_token = nn.Parameter(module.cls_token[:, :, out_mask])
                module.pos_embed = nn.Parameter(module.pos_embed[:, :, out_mask])
            elif isinstance(module, torch.nn.MultiheadAttention):
                in_mask = model.backbone.out_mask.view(-1).bool()
                module.embed_dim = in_mask.sum().item()
                module.head_dim = module.embed_dim // module.num_heads
                proj_weight = module.in_proj_weight[out_mask]
                proj_weight = proj_weight[:, in_mask.view(-1).bool()]
                module.in_proj_weight = nn.Parameter(proj_weight)
                module.in_proj_bias = nn.Parameter(module.in_proj_bias[out_mask])
            else:
                info = []
                if isinstance(module, SqueezeAxialPositionalEmbedding):
                    weight = module.pos_embed
                else:
                    weight = module.weight
                requires_grad = weight.requires_grad

                if hasattr(module, "bias") and module.bias is not None:
                    module.bias = nn.Parameter(module.bias.data[out_mask])
                    if module.bias.grad is not None:
                        module.bias.grad = module.bias.grad[out_mask]
                    module.bias.requires_grad = requires_grad
                oc = len(out_mask)
                if isinstance(module, SqueezeAxialPositionalEmbedding):
                    temp_weight = weight.data[:, out_mask]
                    if weight.grad is not None:
                        temp_weight.grad = weight.grad.data[:, out_mask]
                else:
                    temp_weight = weight.data[out_mask]
                    if weight.grad is not None:
                        temp_weight.grad = weight.grad.data[out_mask]

                if isinstance(module, LayerNorm):
                    module.normalized_shape = (out_mask.sum().item(),)

                info.append(
                    f"{module.name} - Out Channels: {oc} -> {out_mask.sum().item()}"
                )
                if hasattr(module, "out_channels"):
                    module.out_channels = out_mask.sum().item()
                if hasattr(module, "num_features"):
                    module.num_features = out_mask.sum().item()
                if hasattr(module, "in_mask"):
                    in_mask = module.in_mask.view(-1).bool()
                    module.in_mask = module.in_mask[in_mask.view(module.in_mask.shape)]
                    if hasattr(module, "in_channels"):
                        module.in_channels = in_mask.sum().item()
                    if (hasattr(module, "groups") and module.groups == 1) or isinstance(
                        module, nn.Linear
                    ):
                        # only apply if not dwconv
                        try:
                            temp_weight = temp_weight[:, in_mask].data
                            if temp_weight.grad is not None:
                                temp_weight.grad = temp_weight.grad[:, in_mask].data
                        except:
                            print(
                                "Failure",
                                module.name,
                                module.groups,
                                temp_weight.shape,
                                in_mask.shape,
                                out_mask.shape,
                                (
                                    temp_weight.grad.shape
                                    if temp_weight.grad is not None
                                    else None
                                ),
                            )
                            quit()
                        info.append(
                            f" - In Channels: {in_mask.size(0)} -> {in_mask.sum().item()}"
                        )
                    else:
                        assert isinstance(module, torch.nn.Conv2d)
                        module.groups = (
                            in_mask.sum().item() * module.groups
                        ) // in_mask.size(0)
                        info.append(
                            f" - Depthwise Conv {module.groups} {module.in_channels} {in_mask.sum().item()} {temp_weight.shape} {in_mask.shape}"
                        )
                if isinstance(module, SqueezeAxialPositionalEmbedding):
                    module.pos_embed = nn.Parameter(temp_weight)
                    if temp_weight.grad is not None:
                        module.pos_embed.grad = temp_weight.grad
                    module.pos_embed.requires_grad = requires_grad
                else:
                    module.weight = nn.Parameter(temp_weight)
                    if temp_weight.grad is not None:
                        module.weight.grad = temp_weight.grad
                    module.weight.requires_grad = requires_grad
                if isinstance(module, _NormBase):
                    module.running_mean = module.running_mean[out_mask]
                    module.running_var = module.running_var[out_mask]
                module.out_mask = module.out_mask[out_mask.view(module.out_mask.shape)]
                print("".join(info))
            done.append(module)


def add_masks(model: torch.nn.Module):
    param2parent: dict[torch.nn.Parameter, torch.nn.Module] = {}
    for mn, m in model.named_modules():
        m.name = mn
        for _, p in m.named_parameters(prefix=mn, recurse=False):
            param2parent[p] = m
    for p, m in param2parent.items():
        if isinstance(m, Conv2d):
            m.register_buffer(
                "out_mask",
                m.weight.new_ones((m.out_channels, 1, 1, 1)),
            )
            m.register_buffer("in_mask", m.weight.new_ones((1, m.in_channels, 1, 1)))
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
        elif isinstance(m, VisionTransformer):
            if not hasattr(m, "out_mask"):
                m.register_buffer("out_mask", p.new_ones(1, 1, p.size(-1)))
        elif isinstance(m, torch.nn.MultiheadAttention):
            m.register_buffer("out_mask", p.new_ones(3 * m.embed_dim, 1))
            m.register_buffer("in_mask", p.new_ones(1, m.embed_dim))
        else:
            raise TypeError(f"Unknown weight type: {type(m)}")
