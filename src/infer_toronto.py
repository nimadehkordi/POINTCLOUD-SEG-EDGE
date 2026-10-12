"""Run PointNet Toronto_3D inference and write per-point labels as an ASCII PLY.

Example:
    python -m src.infer_toronto --scan data/Toronto_3D/L004.ply \
        --checkpoint outputs/checkpoints/pointnet_toronto.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .model import load_checkpoint
from .toronto_data import (
    NUM_TORONTO_CLASSES,
    TORONTO_CLASS_NAMES,
    load_toronto_scan_for_inference,
    merge_tile_predictions,
)


def write_prediction_ply(source: Path, output: Path, predicted: np.ndarray,
                         n_points: int, tile_size: float, coord_scale: float) -> None:
    from .toronto_data import _read_header, _vertex_layout

    header, offset = _read_header(source)
    count, dtype = _vertex_layout(header)
    records = np.memmap(source, mode="r", dtype=dtype, offset=offset, shape=(count,))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {count}\n")
        stream.write("property double x\nproperty double y\nproperty double z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("property int predicted_label\nend_header\n")
        palette = np.array([
            [128, 128, 128], [70, 180, 75], [245, 220, 40], [30, 150, 60],
            [220, 40, 40], [40, 180, 220], [180, 80, 180], [40, 80, 220], [180, 120, 70],
        ], dtype=np.uint8)
        colors = palette[np.clip(predicted, 0, len(palette) - 1)]
        for i in range(count):
            stream.write(f"{records['x'][i]} {records['y'][i]} {records['z'][i]} "
                         f"{int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])} {int(predicted[i])}\n")
    del records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True, help="Toronto_3D binary PLY")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/pointnet_toronto.pt")
    parser.add_argument("--onnx", default=None, help="Optional ONNX model; defaults to PyTorch checkpoint")
    parser.add_argument("--out", type=Path, default=None, help="Output ASCII PLY with predicted labels")
    parser.add_argument("--tile-size", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model, ckpt = load_checkpoint(args.checkpoint, device="cpu")
    n_points = int(ckpt["n_points"])
    tile_size = args.tile_size or float(ckpt.get("tile_size", 20.0))
    coord_scale = float(ckpt.get("coord_scale", 25.0))
    tiles, point_indices, sampled_labels = load_toronto_scan_for_inference(
        args.scan, n_points=n_points, tile_size=tile_size,
        coord_scale=coord_scale, seed=args.seed,
    )

    if args.onnx:
        session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        logits_list = []
        for start in range(len(tiles)):
            logits_list.append(session.run(None, {"points": tiles[start:start + 1]})[0][0])
        tile_logits = np.stack(logits_list)
    else:
        device = torch.device(args.device if args.device != "auto" else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        ))
        model = model.to(device).eval()
        logits_list = []
        with torch.no_grad():
            for start in range(0, len(tiles), args.batch_size):
                x = torch.from_numpy(tiles[start:start + args.batch_size]).to(device)
                logits_list.extend(model(x).cpu().numpy())
        tile_logits = np.stack(logits_list)

    # (tiles, classes, points) -> (tiles, points, classes) for point-wise voting.
    n_classes = int(ckpt["model_args"]["num_classes"])
    merged = merge_tile_predictions(point_indices, tile_logits.transpose(0, 2, 1),
                                    int(max(indices.max() for indices in point_indices) + 1), n_classes)
    # Every scan point is assigned to a non-overlapping tile, so merge size is the scan size.
    from .toronto_data import _read_header, _vertex_layout
    header, _ = _read_header(args.scan)
    source_count, _ = _vertex_layout(header)
    if len(merged) < source_count:
        merged = np.pad(merged, (0, source_count - len(merged)), constant_values=0)
    if args.out is None:
        args.out = Path("outputs/toronto_predictions") / f"{args.scan.stem}_prediction.ply"
    write_prediction_ply(args.scan, args.out, merged, n_points, tile_size, coord_scale)
    print(f"predicted {source_count:,} points across {len(tiles):,} tiles -> {args.out}")

    if sampled_labels is not None:
        # Evaluate the same model outputs on the sampled labeled points (avoid placeholder classes).
        confusion = np.zeros((NUM_TORONTO_CLASSES, NUM_TORONTO_CLASSES), dtype=np.int64)
        pred = tile_logits.argmax(axis=1)
        np.add.at(confusion, (sampled_labels.reshape(-1), pred.reshape(-1)), 1)
        tp = np.diag(confusion).astype(np.float64)
        union = confusion.sum(0) + confusion.sum(1) - tp
        iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
        print(f"sampled-point OA={tp.sum()/max(confusion.sum(), 1):.3f}, mIoU={np.nanmean(iou):.3f}")
        print("per-class IoU:", {name: (None if np.isnan(v) else round(float(v), 3))
                                 for name, v in zip(TORONTO_CLASS_NAMES, iou)})


if __name__ == "__main__":
    main()