"""Export the trained model to ONNX with a fixed input shape and verify it against PyTorch.

    python -m src.export_onnx

Writes outputs/onnx/pointnet_seg_fp32.onnx. Things to notice in the printout:
* BatchNorm nodes are gone: in eval mode they are folded into the preceding Conv weights,
* every shape is static (batch 1, N points), which is what NPU compilers want,
* the PyTorch vs ONNX Runtime difference should be ~1e-6 (float rounding only).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .data import SyntheticScans
from .model import load_checkpoint
from .qat import to_float_export_model


def export(model: torch.nn.Module, n_points: int, path: str, opset: int = 18) -> None:
    """Prefer the modern torch.export-based exporter (dynamo=True): it captures the graph
    ahead of time and produces a clean static graph. The older TorchScript exporter
    (dynamo=False) also works but leaves shape-computation nodes (ConstantOfShape, Equal,
    Where) around the Expand - a typical example of graph noise you'd remove with
    graph surgery before handing a model to an NPU compiler."""
    model = model.cpu().eval()
    dummy = torch.zeros(1, 4, n_points)
    names = dict(input_names=["points"], output_names=["logits"])
    try:
        torch.onnx.export(model, (dummy,), path, dynamo=True, opset_version=opset, external_data=False, **names)
    except Exception as e:  # e.g. onnxscript missing
        print(f"dynamo exporter failed ({type(e).__name__}: {e}); falling back to the TorchScript exporter")
        torch.onnx.export(model, (dummy,), path, dynamo=False, opset_version=17, do_constant_folding=True, **names)


def describe(path: str) -> None:
    m = onnx.load(path)
    onnx.checker.check_model(m)
    ops = Counter(n.op_type for n in m.graph.node)
    print(f"opset {m.opset_import[0].version}, {len(m.graph.node)} nodes:", dict(ops))
    for io, kind in ((m.graph.input, "input"), (m.graph.output, "output")):
        for v in io:
            dims = [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
            print(f"  {kind}: {v.name} {dims}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/checkpoints/pointnet_seg.pt")
    ap.add_argument("--out", help="output ONNX path (defaults according to checkpoint type)")
    args = ap.parse_args()

    model, ckpt = load_checkpoint(args.checkpoint)
    n_points = ckpt["n_points"]
    out = args.out or ("outputs/onnx/pointnet_seg_qat_fp32.onnx" if ckpt.get("qat")
                       else "outputs/onnx/pointnet_seg_fp32.onnx")
    if ckpt.get("qat"):
        model = to_float_export_model(model)
        print("baked QAT weight fake-quant and BatchNorm; run src.quantize on this float intermediate for QDQ")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    export(model, n_points, out)
    describe(out)

    # Parity check on a few real test scenes.
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    ds = SyntheticScans("test", 5, n_points)
    worst = 0.0
    for i in range(len(ds)):
        x, _ = ds[i]
        ref = model(x[None]).detach().numpy()
        got = sess.run(None, {"points": x[None].numpy()})[0]
        worst = max(worst, float(np.abs(ref - got).max()))
    print(f"max |torch - onnxruntime| over 5 scenes: {worst:.2e}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
