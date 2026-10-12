"""Export a Toronto_3D checkpoint to static-shape ONNX.

Example: python -m src.export_toronto_onnx --checkpoint outputs/checkpoints/pointnet_toronto.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .export_onnx import export, describe
from .model import load_checkpoint
from .toronto_data import load_toronto_scan_for_inference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/checkpoints/pointnet_toronto.pt")
    parser.add_argument("--out", default="outputs/toronto_onnx/pointnet_toronto_fp32.onnx")
    parser.add_argument("--parity-scan", default=None, help="Optional Toronto PLY to use for parity validation")
    args = parser.parse_args()

    model, ckpt = load_checkpoint(args.checkpoint)
    n_points = int(ckpt["n_points"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    export(model, n_points, args.out)
    describe(args.out)

    if args.parity_scan:
        tiles, _, _ = load_toronto_scan_for_inference(
            args.parity_scan, n_points=n_points,
            tile_size=float(ckpt.get("tile_size", 20.0)),
            coord_scale=float(ckpt.get("coord_scale", 25.0)),
        )
        session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        worst = 0.0
        for x in tiles[:5]:
            with torch.no_grad():
                reference = model(torch.from_numpy(x[None])).numpy()
            actual = session.run(None, {"points": x[None]})[0]
            worst = max(worst, float(np.abs(reference - actual).max()))
        print(f"max |PyTorch - ONNX Runtime| across {min(5, len(tiles))} tiles: {worst:.2e}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
