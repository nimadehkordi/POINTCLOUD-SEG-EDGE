"""Post-training static QDQ quantization using representative Toronto_3D tiles.

Example: python -m src.quantize_toronto --calib-scans L001.ply L002.ply
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from .quantize import variant_configs
from .toronto_data import load_toronto_scan_for_inference


class TorontoCalibrationReader(CalibrationDataReader):
    def __init__(self, scans: list[Path], n_points: int, tile_size: float,
                 coord_scale: float, max_tiles: int, seed: int = 0):
        tiles = []
        for scan_i, scan in enumerate(scans):
            features, _, _ = load_toronto_scan_for_inference(
                scan, n_points=n_points, tile_size=tile_size,
                coord_scale=coord_scale, seed=seed + scan_i,
            )
            tiles.extend(features)
            if len(tiles) >= max_tiles:
                break
        if not tiles:
            raise ValueError("Calibration scans did not produce any tiles")
        self.samples = [x[None].astype(np.float32, copy=False) for x in tiles[:max_tiles]]
        if not all(np.isfinite(x).all() for x in self.samples):
            raise ValueError("Calibration data contain non-finite input values")
        self.rewind()

    def get_next(self):
        if self._index >= len(self.samples):
            return None
        sample = self.samples[self._index]
        self._index += 1
        return {"points": sample}

    def rewind(self):
        self._index = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", default="outputs/toronto_onnx/pointnet_toronto_fp32.onnx")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/pointnet_toronto.pt")
    parser.add_argument("--data-dir", type=Path, default=Path("data/Toronto_3D"))
    parser.add_argument("--calib-scans", nargs="+", default=["L001.ply", "L002.ply"])
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--calib-tiles", type=int, default=64)
    parser.add_argument("--out-dir", default="outputs/toronto_onnx")
    args = parser.parse_args()

    fp32 = Path(args.fp32)
    if not fp32.is_file():
        parser.error(f"FP32 ONNX model not found: {fp32}; export it first")
    checkpoint = __import__("torch").load(args.checkpoint, map_location="cpu", weights_only=False)
    n_points = int(checkpoint["n_points"])
    tile_size = float(checkpoint.get("tile_size", 20.0))
    coord_scale = float(checkpoint.get("coord_scale", 25.0))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prep = out_dir / "pointnet_toronto_fp32_prep.onnx"
    quant_pre_process(str(fp32), str(prep), skip_symbolic_shape=True)
    print(f"pre-processed float model -> {prep}")

    scans = [Path(s) if Path(s).is_absolute() else args.data_dir / s for s in args.calib_scans]
    missing = [str(s) for s in scans if not s.is_file()]
    if missing:
        parser.error(f"Calibration scan(s) not found: {', '.join(missing)}")
    reader = TorontoCalibrationReader(scans, n_points, tile_size, coord_scale, args.calib_tiles)
    configs = variant_configs(prep)
    names = args.variants or list(configs)
    for name in names:
        if name not in configs:
            parser.error(f"Unknown variant {name!r}; choose from {', '.join(configs)}")
        cfg = dict(configs[name])
        extra = {"ActivationSymmetric": False, "WeightSymmetric": True, **cfg.pop("extra_options", {})}
        out = out_dir / f"pointnet_toronto_{name}.onnx"
        reader.rewind()
        start = time.time()
        quantize_static(str(prep), str(out), reader, quant_format=QuantFormat.QDQ,
                        extra_options=extra, **cfg)
        q_nodes = sum(node.op_type == "QuantizeLinear" for node in onnx.load(str(out)).graph.node)
        print(f"{name:18s} -> {out} {out.stat().st_size / 1024:.0f} KB; {q_nodes} Q nodes ({time.time()-start:.1f}s)")


if __name__ == "__main__":
    main()
