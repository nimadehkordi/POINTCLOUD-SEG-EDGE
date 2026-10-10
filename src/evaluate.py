"""Evaluate accuracy and latency of the PyTorch model and every ONNX model side by side.

    python -m src.evaluate                  # writes outputs/results.csv

Accuracy: mIoU / overall accuracy on held-out test scenes.
Latency: ONNX Runtime CPU, batch 1, after warm-up; p50 and p99 over many runs.

Note on latency: on a laptop CPU, ONNX Runtime may run small int8 models only a little
faster than float, or even slower, because quantize/dequantize overhead dominates for
small layers. The big speed-ups of integer inference appear on hardware with int8 matrix
units (NPUs, DSPs, Tensor Cores). Accuracy numbers, however, transfer well.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .data import NUM_CLASSES, SyntheticScans
from .metrics import confusion_matrix, summarize
from .model import load_checkpoint


def make_session(path: str, optimize: bool = True, threads: int = 0) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimize else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def eval_onnx(path: str, n_scenes: int = 100, n_points: int = 2048) -> dict:
    sess = make_session(path)
    ds = SyntheticScans("test", n_scenes, n_points)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for i in range(n_scenes):
        x, y = ds[i]
        logits = sess.run(None, {"points": x[None].numpy()})[0]
        cm += confusion_matrix(logits.argmax(1), y.numpy()[None])
    return summarize(cm)


@torch.no_grad()
def eval_torch(checkpoint: str, n_scenes: int = 100, n_points: int = 2048) -> dict:
    model, _ = load_checkpoint(checkpoint)
    ds = SyntheticScans("test", n_scenes, n_points)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for i in range(n_scenes):
        x, y = ds[i]
        cm += confusion_matrix(model(x[None]).argmax(1).numpy(), y.numpy()[None])
    return summarize(cm)


def benchmark(path: str, n_points: int = 2048, warmup: int = 20, iters: int = 200, threads: int = 1) -> dict:
    """Single-threaded by default so results are stable and comparable between models."""
    sess = make_session(path, threads=threads)
    x = SyntheticScans("test", 1, n_points)[0][0][None].numpy()
    for _ in range(warmup):
        sess.run(None, {"points": x})
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {"points": x})
        times.append((time.perf_counter() - t0) * 1e3)
    return {"p50_ms": float(np.percentile(times, 50)), "p99_ms": float(np.percentile(times, 99))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/checkpoints/pointnet_seg.pt")
    ap.add_argument("--onnx-dir", default="outputs/onnx")
    ap.add_argument("--n-scenes", type=int, default=100)
    ap.add_argument("--out", default="outputs/results.csv")
    args = ap.parse_args()

    n_points = torch.load(args.checkpoint, weights_only=False)["n_points"]
    rows = [{"model": "pytorch_fp32", **eval_torch(args.checkpoint, args.n_scenes, n_points)}]
    for path in sorted(Path(args.onnx_dir).glob("pointnet_seg_*.onnx")):
        if path.stem.endswith("_prep"):
            continue
        row = {"model": path.stem.replace("pointnet_seg_", "onnx_"), **eval_onnx(str(path), args.n_scenes, n_points)}
        row.update(benchmark(str(path), n_points))
        row["size_kb"] = round(path.stat().st_size / 1024)
        rows.append(row)

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(f"{'model':22s} {'mIoU':>6s} {'OA':>6s} {'p50 ms':>7s} {'p99 ms':>7s} {'KB':>6s}")
    for r in rows:
        print(f"{r['model']:22s} {r['mIoU']:6.3f} {r['OA']:6.3f} {r.get('p50_ms', float('nan')):7.2f} "
              f"{r.get('p99_ms', float('nan')):7.2f} {r.get('size_kb', ''):>6}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
