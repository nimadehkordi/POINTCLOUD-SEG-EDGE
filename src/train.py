"""Train the PointNet segmentation model on synthetic scans.

    python -m src.train                    # default: ~5-10 min on an Apple Silicon Mac (MPS)
    python -m src.train --epochs 3 --train-size 400   # quick smoke test
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

from .data import NUM_CLASSES, SyntheticScans, class_frequencies
from .metrics import confusion_matrix, summarize
from .model import PointNetSeg, count_macs, count_parameters, load_checkpoint, pick_device
from .qat import prepare_qat_model


@torch.no_grad()
def evaluate_torch(model, loader, device) -> dict:
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        cm += confusion_matrix(pred, y.numpy())
    return summarize(cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train-size", type=int, default=2000, help="scenes per epoch")
    ap.add_argument("--val-size", type=int, default=200)
    ap.add_argument("--n-points", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="auto", help="auto | mps | cpu | cuda")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="outputs/checkpoints/pointnet_seg.pt")
    ap.add_argument("--qat", action="store_true", help="train with W8A8 fake quantization (QAT)")
    ap.add_argument("--init-checkpoint", help="optional float or QAT checkpoint to fine-tune")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu") if args.qat and args.device == "auto" else pick_device(args.device)
    model_args = dict(in_channels=4, num_classes=NUM_CLASSES)
    initial_ckpt = None
    if args.init_checkpoint:
        model, initial_ckpt = load_checkpoint(args.init_checkpoint, device="cpu")
        model_args = initial_ckpt["model_args"]
        if bool(initial_ckpt.get("qat")) and not args.qat:
            ap.error("a QAT checkpoint can only be resumed with --qat")
    else:
        model = PointNetSeg(**model_args)
    if args.qat and not (initial_ckpt and bool(initial_ckpt.get("qat"))):
        model = prepare_qat_model(model)
    model = model.to(device)
    print(f"device={device}  params={count_parameters(model)/1e3:.0f}k  MACs/scene={count_macs(model, args.n_points)/1e6:.0f}M")
    if args.qat:
        print("QAT enabled: W8A8, unsigned per-tensor activations, symmetric per-channel weights")

    train_ds = SyntheticScans("train", args.train_size, args.n_points)
    val_ds = SyntheticScans("val", args.val_size, args.n_points)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                          persistent_workers=args.workers > 0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    # Inverse-sqrt-frequency class weights: rare classes (poles, noise) matter more.
    freq = class_frequencies(100, args.n_points)
    weights = torch.tensor(1.0 / np.sqrt(freq + 1e-6), dtype=torch.float32)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    print("class weights:", np.round(weights.numpy(), 2))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.parent.parent / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_mIoU", "val_OA", "seconds"])

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        val = evaluate_torch(model, val_dl, device)
        dt = time.time() - t0
        print(f"epoch {epoch:3d}  loss {np.mean(losses):.4f}  val mIoU {val['mIoU']:.3f}  OA {val['OA']:.3f}  ({dt:.0f}s)")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, np.mean(losses), val["mIoU"], val["OA"], dt])
        if val["mIoU"] > best:
            best = val["mIoU"]
            torch.save({"state_dict": model.state_dict(), "model_args": model_args, "n_points": args.n_points,
                        "epoch": epoch, "val": val,
                        "qat": {"enabled": True, "scheme": "w8a8_perchannel"} if args.qat else False}, out)

    print(f"best val mIoU {best:.3f}; checkpoint -> {out}; log -> {log_path}")


if __name__ == "__main__":
    main()
