import argparse
from pathlib import Path
from prune.prune_cfg import PruneCfg
from prune.tools import deploy_pruning, add_masks

import torch
from torch import nn
from mmengine.runner import load_checkpoint, save_checkpoint
from mmengine.logging import MMLogger
from mmengine import Config
from mmengine.registry import init_default_scope
from mmseg.models import BaseSegmentor
from mmseg.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(description="Export pruned model to onnx")
    parser.add_argument("config", help="Model export config")
    parser.add_argument("out", help="Export path")
    parser.add_argument(
        "--gen-in-mask", action="store_true", help="Generate the in-masks [legacy]"
    )
    parser.add_argument("--prune-cfg", help="Pruning config file path")
    parser.add_argument("--pth", help="Pth checkpoint")
    parser.add_argument("--scope", help="MMRegistry Scope")
    parser.add_argument(
        "--shape", type=int, nargs="+", default=[512, 512], help="input image size"
    )
    parser.add_argument(
        "--gen-pth",
        action="store_true",
        help="Generate .pth checkpoint instead of onnx",
    )
    parser.add_argument(
        "--print-complexity",
        action="store_true",
        help="Print complexity information of the pruned model",
    )
    parser.add_argument(
        "--skip-pruning",
        action="store_true",
        help="Skip application of pruned masks before exporting",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name="MMLogger")
    if len(args.shape) == 1:
        h, w = args.shape[0], args.shape[0]
    elif len(args.shape) == 2:
        h, w = args.shape
    else:
        raise ValueError("invalid input shape")

    config_name = Path(args.config)

    if not config_name.exists():
        logger.error(f"Config file {config_name} does not exist")

    cfg: Config = Config.fromfile(config_name)
    # cfg.log_level = "WARN"

    scope = "mmseg"
    if args.scope:
        scope = args.scope
    init_default_scope(cfg.get("scope", scope))

    model: BaseSegmentor = MODELS.build(cfg.model)
    if not args.skip_pruning:
        # Add masks to prevent unexpected keys warning when loading checkpoint
        add_masks(model)
    checkpoint = None
    if checkpoint in cfg:
        checkpoint = cfg.checkpoint
    elif args.pth:
        checkpoint = args.pth

    if checkpoint is not None:
        load_checkpoint(model, checkpoint, revise_keys=[(r"^architecture\.", "")])

    if args.gen_in_mask:
        # Debugging
        prune_cfg_name = Path(args.prune_cfg)

        if not prune_cfg_name.exists():
            logger.error(f"Prune config file {prune_cfg_name} does not exist")
        logger.info("Generating in masks")
        gen_in_masks(model, prune_cfg_name)

    if not args.skip_pruning:
        logger.info("Deploying pruning")
        deploy_pruning(model)
    if args.print_complexity:
        from ptflops import get_model_complexity_info

        macs, params = get_model_complexity_info(
            model, (3, h, w), as_strings=True, print_per_layer_stat=True, verbose=True
        )
        print("{:<30}  {:<8}".format("Computational complexity: ", macs))
        print("{:<30}  {:<8}".format("Number of parameters: ", params))
        quit()
    model = model.cuda()
    inputs = torch.zeros(1, 3, h, w).cuda()
    model.eval()
    out = model(inputs)
    if not args.gen_pth:
        import onnx

        torch.onnx.export(
            model, inputs, args.out, input_names=["input"], output_names=["output"]
        )
    else:
        save_checkpoint(model.state_dict(), args.out)


def gen_in_masks(model: nn.Module, prune_cfg_path: Path):
    prune_cfg = PruneCfg(file=prune_cfg_path)

    modules = {}
    for n, m in model.named_modules():
        m.name = n
        modules[n] = m

    def get_base_op(op):
        split = op.split(".")
        if split[-1] == "weight" or split[-1] == "pos_embed":
            split = split[:-1]
        # if split[-1] == "c" or split[-1] == "conv":
        #     split = split[:-1]
        return ".".join(split)

    for g in prune_cfg.groups:
        og, in_group = g["out_group"], g["out_in_group"]
        mask = modules[get_base_op(og[0])].out_mask
        for ig in in_group:
            modules[get_base_op(ig)].in_mask = mask.transpose(0, 1)


if __name__ == "__main__":
    main()
