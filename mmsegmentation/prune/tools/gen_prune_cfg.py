# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from pathlib import Path
import tempfile

import torch

from mmengine import Config
from mmseg.registry import MODELS
from mmseg.models import BaseSegmentor
from mmseg.structures import SegDataSample
from mmengine.registry import init_default_scope
from prune.prune_cfg import PruneCfg
from prune.structure_analyzer import StructureAnalyzer


from ptflops import get_model_complexity_info


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pruning config")
    parser.add_argument("config", help="config file path")
    parser.add_argument("--out", help="Output config file path")
    parser.add_argument("--scope", help="MMRegistry Scope")
    parser.add_argument(
        "--shape", type=int, nargs="+", default=[512, 512], help="input image size"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if len(args.shape) == 1:
        h, w = args.shape[0], args.shape[0]
    elif len(args.shape) == 2:
        h, w = args.shape
    else:
        raise ValueError("invalid input shape")

    config_name = Path(args.config)

    cfg = Config.fromfile(config_name)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = "WARN"
    scope = "mmseg"
    if args.scope:
        scope = args.scope
    init_default_scope(cfg.get("scope", scope))
    model: BaseSegmentor = MODELS.build(cfg.model).cuda()
    result = {}
    result["ori_shape"] = (h, w)
    result["pad_shape"] = (h, w)
    data_batch = {
        "inputs": [torch.rand(3, h, w)],
        "data_samples": [SegDataSample(metainfo=result)],
    }
    data = model.data_preprocessor(data_batch)

    model.eval()
    sa = StructureAnalyzer(model)
    sa.generate_structure(model, inputs=data["inputs"])

    prune_cfg = PruneCfg(structure=sa)

    if args.out:
        out_path = args.out
    elif cfg.prune_cfg:
        out_path = cfg.prune_cfg
    else:
        raise ValueError(
            "Either prune_cfg path in the model config or --out have to be specified"
        )
    prune_cfg.save_to_file(out_path)


if __name__ == "__main__":
    main()
