"""Train PointNet on cached spatial tiles from Toronto_3D scans.

Example:
    python -m src.train_toronto --data-dir data/Toronto_3D \
        --train-scans L001.ply L002.ply L003.ply --val-scans L004.ply
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import PointNetSeg, count_macs, count_parameters, pick_device
from .toronto_data import NUM_TORONTO_CLASSES, TORONTO_CLASS_NAMES, Toronto3DTiles


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, np.ndarray]:
    model.eval()
    confusion = np.zeros((NUM_TORONTO_CLASSES, NUM_TORONTO_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for x, target in loader:
            prediction = model(x.to(device)).argmax(1).cpu().numpy()
            target = target.numpy()
            encoded = target.reshape(-1) * NUM_TORONTO_CLASSES + prediction.reshape(-1)
            confusion += np.bincount(encoded, minlength=NUM_TORONTO_CLASSES**2).reshape(
                NUM_TORONTO_CLASSES, NUM_TORONTO_CLASSES
            )

    true_positive = np.diag(confusion).astype(np.float64)
    denominator = confusion.sum(0) + confusion.sum(1) - true_positive
    iou = np.divide(true_positive, denominator, out=np.full_like(true_positive, np.nan), where=denominator > 0)
    metrics = {
        "mIoU": float(np.nanmean(iou)),
        "OA": float(true_positive.sum() / max(confusion.sum(), 1)),
    }
    metrics.update({f"IoU_{name}": float(value) for name, value in zip(TORONTO_CLASS_NAMES, iou)})
    return metrics, confusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/Toronto_3D"))
    parser.add_argument("--train-scans", nargs="+", default=["L001.ply", "L002.ply", "L003.ply"])
    parser.add_argument("--val-scans", nargs="+", default=["L004.ply"])
    parser.add_argument("--tile-size", type=float, default=20.0, help="XY tile width in metres")
    parser.add_argument("--min-points", type=int, default=2048, help="drop tiles below this source-point count")
    parser.add_argument("--n-points", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto | mps | cpu | cuda")
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/toronto_cache"))
    parser.add_argument("--out", type=Path, default=Path("outputs/checkpoints/pointnet_toronto.pt"))
    args = parser.parse_args()

    train_paths = [args.data_dir / name for name in args.train_scans]
    val_paths = [args.data_dir / name for name in args.val_scans]
    if {path.resolve() for path in train_paths} & {path.resolve() for path in val_paths}:
        parser.error("train and validation scans must be disjoint to prevent spatial leakage")

    torch.manual_seed(0)
    np.random.seed(0)
    device = pick_device(args.device)
    train_ds = Toronto3DTiles(train_paths, n_points=args.n_points, tile_size=args.tile_size,
                              min_points=args.min_points, augment=True, cache_dir=args.cache_dir, seed=10)
    val_ds = Toronto3DTiles(val_paths, n_points=args.n_points, tile_size=args.tile_size,
                            min_points=args.min_points, augment=False, cache_dir=args.cache_dir, seed=20)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=False,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, persistent_workers=args.workers > 0)

    model_args = {"in_channels": 4, "num_classes": NUM_TORONTO_CLASSES}
    model = PointNetSeg(**model_args).to(device)
    print(f"device={device} params={count_parameters(model)/1e3:.0f}k "
          f"MACs/tile={count_macs(model, args.n_points)/1e6:.0f}M")
    print(f"tiles: train={len(train_ds):,}, val={len(val_ds):,}; "
          f"classes={NUM_TORONTO_CLASSES} (including Unclassified)")

    frequencies = train_ds.class_counts.astype(np.float64)
    frequencies /= frequencies.sum()
    weights = 1.0 / np.sqrt(np.maximum(frequencies, 1e-6))
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    print("class weights:", {name: round(float(weight), 2)
                              for name, weight in zip(TORONTO_CLASS_NAMES, weights)})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.1
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_dir = args.out.parent.parent if args.out.parent.name == "checkpoints" else args.out.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "toronto_train_log.csv"
    with log_path.open("w", newline="") as stream:
        csv.writer(stream).writerow(["epoch", "train_loss", "val_mIoU", "val_OA", "seconds"])

    best_miou = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time = time.time()
        losses = []
        for features, target in train_loader:
            if not torch.isfinite(features).all():
                raise ValueError("Non-finite feature values in Toronto_3D batch; clear outputs/toronto_cache and rebuild it")
            features, target = features.to(device), target.to(device)
            loss = criterion(model(features), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))

        metrics, _ = evaluate(model, val_loader, device)
        elapsed = time.time() - start_time
        train_loss = float(np.mean(losses))
        print(f"epoch {epoch:3d} loss {train_loss:.4f} val mIoU {metrics['mIoU']:.3f} "
              f"OA {metrics['OA']:.3f} ({elapsed:.0f}s)")
        with log_path.open("a", newline="") as stream:
            csv.writer(stream).writerow([epoch, train_loss, metrics["mIoU"], metrics["OA"], elapsed])
        if metrics["mIoU"] > best_miou:
            best_miou = metrics["mIoU"]
            torch.save({
                "state_dict": model.state_dict(),
                "model_args": model_args,
                "n_points": args.n_points,
                "dataset": "Toronto_3D",
                "class_names": TORONTO_CLASS_NAMES,
                "tile_size": args.tile_size,
                "coord_scale": 25.0,
                "train_scans": args.train_scans,
                "val_scans": args.val_scans,
                "epoch": epoch,
                "val": metrics,
            }, args.out)
    print(f"best val mIoU {best_miou:.3f}; checkpoint -> {args.out}; log -> {log_path}")


if __name__ == "__main__":
    main()
