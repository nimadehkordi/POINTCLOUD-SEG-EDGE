"""Post-training static quantization (PTQ) of the ONNX model with ONNX Runtime.

    python -m src.quantize                       # all variants
    python -m src.quantize --variants w8a8_perchannel w8a16

Pipeline:
1. quant_pre_process: shape inference + graph cleanup -> pointnet_seg_fp32_prep.onnx
   (this is the float reference the quantized models are compared against),
2. calibration: run ~64 validation scenes through the model and record activation ranges,
3. quantize_static in QDQ format: QuantizeLinear/DequantizeLinear pairs are inserted around
   tensors. Runtimes such as ORT, TensorRT and QNN read these pairs and run integer kernels.

Variants (all PTQ, no retraining):
  w8a8_pertensor      8-bit weights (one scale per tensor), 8-bit activations, min-max
  w8a8_perchannel     8-bit weights with one scale per output channel, min-max
  w8a8_percentile     per-channel + percentile calibration (clips activation outliers)
  w8a8_entropy        per-channel + entropy (KL) calibration
  w8a8_head_fp32      per-channel, but the final classifier Conv stays in float
  w8a16               8-bit per-channel weights, 16-bit activations (like many NPU deployments)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from .data import SyntheticScans

ONNX_DIR = Path("outputs/onnx")
FP32 = ONNX_DIR / "pointnet_seg_fp32.onnx"
FP32_PREP = ONNX_DIR / "pointnet_seg_fp32_prep.onnx"


class ScanCalibrationReader(CalibrationDataReader):
    """Feeds calibration scenes to the quantizer, one at a time (batch 1, like the deployed model)."""

    def __init__(self, n_scenes: int = 64, n_points: int = 2048, input_name: str = "points"):
        ds = SyntheticScans("val", n_scenes, n_points, augment_data=False)
        self.samples = [ds[i][0][None].numpy() for i in range(n_scenes)]
        self.input_name = input_name
        self.rewind()

    def get_next(self):
        if self._i >= len(self.samples):
            return None
        self._i += 1
        return {self.input_name: self.samples[self._i - 1]}

    def rewind(self):
        self._i = 0


def last_conv_name(model_path: Path) -> str:
    convs = [n.name for n in onnx.load(str(model_path)).graph.node if n.op_type == "Conv"]
    return convs[-1]


def variant_configs(prep_model: Path) -> dict:
    a8, a16, w8 = QuantType.QUInt8, QuantType.QUInt16, QuantType.QInt8
    return {
        "w8a8_pertensor": dict(per_channel=False, activation_type=a8, weight_type=w8, calibrate_method=CalibrationMethod.MinMax),
        "w8a8_perchannel": dict(per_channel=True, activation_type=a8, weight_type=w8, calibrate_method=CalibrationMethod.MinMax),
        "w8a8_percentile": dict(per_channel=True, activation_type=a8, weight_type=w8, calibrate_method=CalibrationMethod.Percentile,
                                extra_options={"CalibPercentile": 99.99}),
        "w8a8_entropy": dict(per_channel=True, activation_type=a8, weight_type=w8, calibrate_method=CalibrationMethod.Entropy),
        "w8a8_head_fp32": dict(per_channel=True, activation_type=a8, weight_type=w8, calibrate_method=CalibrationMethod.MinMax,
                               nodes_to_exclude=[last_conv_name(prep_model)]),
        "w8a16": dict(per_channel=True, activation_type=a16, weight_type=w8, calibrate_method=CalibrationMethod.MinMax,
                      extra_options={"UseQDQContribOps": True}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", default=str(FP32))
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--calib-scenes", type=int, default=64)
    ap.add_argument("--n-points", type=int, default=None,
                    help="points per scene (defaults to checkpoint metadata for a recognized model)")
    ap.add_argument("--name-prefix", default="pointnet_seg",
                    help="prefix for generated ONNX files, e.g. pointnet_seg_qat")
    args = ap.parse_args()

    prep_model = Path(args.fp32).with_name(f"{Path(args.fp32).stem}_prep.onnx")
    quant_pre_process(args.fp32, str(prep_model), skip_symbolic_shape=True)
    print(f"pre-processed float model -> {prep_model}")

    configs = variant_configs(prep_model)
    names = args.variants or list(configs)
    n_points = args.n_points
    if n_points is None:
        input_shape = onnx.load(args.fp32).graph.input[0].type.tensor_type.shape.dim
        n_points = input_shape[2].dim_value if len(input_shape) > 2 else 0
        if n_points <= 0:
            n_points = 2048
    reader = ScanCalibrationReader(args.calib_scenes, n_points)
    for name in names:
        cfg = dict(configs[name])
        extra = {"ActivationSymmetric": False, "WeightSymmetric": True, **cfg.pop("extra_options", {})}
        out = Path(args.fp32).parent / f"{args.name_prefix}_{name}.onnx"
        reader.rewind()
        t0 = time.time()
        quantize_static(str(prep_model), str(out), reader, quant_format=QuantFormat.QDQ, extra_options=extra, **cfg)
        n_q = sum(n.op_type == "QuantizeLinear" for n in onnx.load(str(out)).graph.node)
        size_kb = out.stat().st_size / 1024
        print(f"{name:18s} -> {out.name:36s} {size_kb:7.0f} KB  {n_q:3d} QuantizeLinear nodes  ({time.time()-t0:.1f}s)")
    fp_kb = Path(args.fp32).stat().st_size / 1024
    print(f"(float model: {fp_kb:.0f} KB. QDQ files store int8 weights, so they shrink ~4x for weight-heavy models.)")


if __name__ == "__main__":
    main()
