import argparse
from pathlib import Path
from prune.prune_cfg import PruneCfg


import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.runner import load_checkpoint
from mmengine.logging import MMLogger
from mmengine import Config, DictAction
from mmengine.registry import init_default_scope
from mmseg.models import BaseSegmentor
from mmseg.registry import MODELS

import onnxruntime
import timeit
import numpy


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark onnx model")
    parser.add_argument("path", help="Onnx model path")
    parser.add_argument(
        "--shape", type=int, nargs="+", default=[512, 512], help="input image size"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name="MMLogger")

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (
            1,
            3,
        ) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")

    session = onnxruntime.InferenceSession(
        args.path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_sample = numpy.random.randn(*input_shape).astype(numpy.float32)
    input_name = session.get_inputs()[0].name
    inputs = {input_name: input_sample}
    session.run(None, inputs)
    timeit.repeat(lambda: session.run(None, inputs), number=1, repeat=0)  # Dry run
    latency_list = timeit.repeat(
        lambda: session.run(None, inputs), number=1, repeat=1000
    )
    res = get_latency_result(latency_list, 1)
    print(f"Evaluating {args.path}")
    for r in res.items():
        print(r)


def get_latency_result(latency_list, batch_size):
    latency_ms = sum(latency_list) / float(len(latency_list)) * 1000.0
    latency_variance = numpy.var(latency_list, dtype=numpy.float64) * 1000.0
    throughput = batch_size * (1000.0 / latency_ms)

    return {
        "test_times": len(latency_list),
        "latency_variance": f"{latency_variance:.2f}",
        "latency_90_percentile": f"{numpy.percentile(latency_list, 90) * 1000.0:.2f}",
        "latency_95_percentile": f"{numpy.percentile(latency_list, 95) * 1000.0:.2f}",
        "latency_99_percentile": f"{numpy.percentile(latency_list, 99) * 1000.0:.2f}",
        "min_ms": f"{numpy.min(latency_list) * 1000.0:.2f}",
        "average_latency_ms": f"{latency_ms:.2f}",
        "QPS": f"{throughput:.2f}",
    }


if __name__ == "__main__":
    main()
