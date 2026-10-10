# Point-cloud segmentation: train → ONNX → quantize → inspect

A small, laptop-sized version of what an embedded AI team at a measurement company does: segment
laser-scan point clouds into classes, then get the model ready for an on-device accelerator.

```
synthetic laser scans ─► PointNet segmentation (PyTorch) ─► ONNX (static shapes, BN folded)
      ─► PTQ variants (W8A8 per-tensor / per-channel / percentile / entropy, W8A16, float head)
      ─► accuracy + latency table ─► layer-by-layer SQNR vs float
```

Everything runs on a Mac in minutes. No dataset download is needed: scenes are generated
procedurally with realistic scanner effects (1/r² density, range noise, outliers, intensity).

## Setup (once)

```bash
cd pointcloud-seg-edge
./setup.sh                       # creates .venv, installs requirements, registers a Jupyter kernel
source .venv/bin/activate
```

## Run

```bash
QUICK=1 ./run_all.sh             # ~2-3 min smoke test
./run_all.sh                     # full run: 20 epochs, ~5-10 min on Apple Silicon (MPS)
jupyter lab notebooks/           # pick the "Python (pointcloud-seg-edge)" kernel
```

Or step by step:

| Step | Command | Output |
| --- | --- | --- |
| Train | `python -m src.train` | `outputs/checkpoints/pointnet_seg.pt`, `outputs/train_log.csv` |
| Export | `python -m src.export_onnx` | `outputs/onnx/pointnet_seg_fp32.onnx` + parity check vs PyTorch |
| Quantize | `python -m src.quantize` | `outputs/onnx/pointnet_seg_<variant>.onnx` |
| Evaluate | `python -m src.evaluate` | `outputs/results.csv` (mIoU, per-class IoU, p50/p99 latency, size) |
| Compare layers | `python -m src.compare_layers` | `outputs/layer_compare/*.csv`, `layer_sqnr.png` |
| Pretrained model (optional) | `python -m src.pretrained_image_seg --image photo.jpg` | downloads LR-ASPP MobileNetV3 (~13 MB), exports, quantizes, compares |

Training picks the Apple GPU (`mps`) automatically; force CPU with `--device cpu`.
Optional W8A8 QAT fine-tuning starts from the float checkpoint and writes a separate artifact:

```bash
python -m src.train --qat --init-checkpoint outputs/checkpoints/pointnet_seg.pt \
  --out outputs/checkpoints/pointnet_seg_qat.pt --epochs 5 --lr 2e-4
python -m src.export_onnx --checkpoint outputs/checkpoints/pointnet_seg_qat.pt
```

QAT runs on CPU by default (pass `--device` explicitly to override). It uses unsigned per-tensor
activation fake quantization and symmetric per-output-channel int8 weight fake quantization. The
QAT checkpoint is kept separate from the float checkpoint; ONNX export defaults to
`outputs/onnx/pointnet_seg_qat_fp32.onnx`, baking its learned weight fake quantization and folded
BatchNorm into a float ONNX intermediate. Run `python -m src.quantize --fp32
outputs/onnx/pointnet_seg_qat_fp32.onnx --name-prefix pointnet_seg_qat` to produce separate
deployable QDQ variants from QAT weights without overwriting the baseline PTQ files. The activation
fake-quant behavior is used during training; static ONNX calibration creates the final QDQ
activation nodes.

## Notebooks

* **`01_point_cloud_data.ipynb`**: what scans look like (3D, top-down, side views), class balance, density and
  noise vs range, intensity, the model's input tensor, permutation invariance, voxel downsampling,
  range-image projection, and predictions of the trained model with a confusion matrix.
* **`02_quantization_deep_dive.ipynb`**: scale/zero-point by hand, symmetric vs asymmetric, why to normalize
  inputs, BatchNorm folding, per-tensor vs per-channel weights, activation histograms and the clipping/rounding
  trade-off, reading a QDQ ONNX graph, accuracy/latency of every variant, and a layer-by-layer SQNR plot.

## Project layout

```
src/data.py                 synthetic scan generator + Dataset (6 classes)
src/model.py                PointNet segmentation (Conv1d/BN/ReLU, max-pool, concat; no kNN)
src/train.py                AdamW + OneCycle, class-weighted CE, best-by-val-mIoU checkpoint
src/export_onnx.py          torch.onnx export (opset 18, static shapes) + ORT parity check
src/quantize.py             ONNX Runtime static PTQ, QDQ format, 6 variants
src/evaluate.py             mIoU / per-class IoU / latency for PyTorch and every ONNX file
src/compare_layers.py       expose intermediate tensors, SQNR + cosine per layer
src/pretrained_image_seg.py optional: same loop on a downloaded MobileNetV3 segmentation model
```

## Results from a short CPU run (8 epochs)

| model | mIoU | p50 ms (1 CPU thread) |
| --- | --- | --- |
| PyTorch / ONNX fp32 | 0.828 | 5.6 |
| W8A8 per-tensor | 0.824 | 2.0 |
| W8A8 per-channel | 0.827 | 1.9 |
| W8A16 | 0.827 | 6.9* |

\*ONNX Runtime runs 16-bit QDQ with generic kernels on CPU; NPUs with native 16-bit support (e.g. Qualcomm HTP) don't have this penalty.
Your numbers will differ slightly after the full training run.

## Using real data later

The model only needs `(4, N)` inputs (x, y, z, intensity) and per-point labels. Good public datasets with similar
content: **Toronto-3D** and **Paris-Lille-3D** (mobile laser scanning, urban), **Semantic3D** (terrestrial scanners,
closest to tripod scanners), **S3DIS** (indoor). Write a `Dataset` that tiles each scan into blocks, samples N points
per block, and normalizes coordinates per block, then reuse every other script unchanged.

## Ideas to extend

* Quantization-aware training (QAT) for 4-bit weights.
* Replace `Concat → Conv` with two convs and a broadcast add (the global half then runs once per scene, not per point).
* Export with the TorchScript exporter (`dynamo=False`) and remove the extra shape nodes with graph surgery (`onnx` API).
* Tiling + overlap merge for large scans; benchmark with 4096 and 8192 points.
