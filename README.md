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

## Toronto_3D data and training

Download the Toronto_3D dataset from https://github.com/WeikaiTan/Toronto-3D/tree/master and place its `Toronto_3D` folder under `data/`. Use [notebooks/03_toronto_3d_eda.ipynb](notebooks/03_toronto_3d_eda.ipynb) to explore the scans.

The real-data loader divides each scan into non-overlapping XY tiles, samples a fixed number of points per tile, and supplies normalized XYZ plus intensity and a semantic class ID per point. Non-finite intensity readings are replaced with zero during preprocessing. The first run creates reusable arrays under `outputs/toronto_cache/`; subsequent runs reuse them. Training and validation scans are separate by default to avoid evaluating on neighboring tiles from a scan used in training:

```bash
python -m src.train_toronto --data-dir data/Toronto_3D \
  --train-scans L001.ply L002.ply L003.ply --val-scans L004.ply
```

To do a quick pipeline check, try fewer epochs and a smaller point count first. The checkpoint is written to `outputs/checkpoints/pointnet_toronto.pt`. Toronto_3D has nine labels, including `Unclassified`, so its model head is configured for nine classes and is separate from the synthetic-data checkpoint.

### Toronto_3D inference and PTQ

Export the Toronto checkpoint to a static-shape ONNX model, optionally checking PyTorch/ONNX parity on a scan:

```bash
python -m src.export_toronto_onnx \
  --checkpoint outputs/checkpoints/pointnet_toronto.pt \
  --parity-scan data/Toronto_3D/L004.ply
```

Run inference on a PLY scan using either the PyTorch checkpoint or an ONNX model. The output is an ASCII PLY with predicted labels and class colors; if the input contains semantic labels, sampled-tile OA and mIoU are also reported.

```bash
python -m src.infer_toronto --scan data/Toronto_3D/L004.ply
python -m src.infer_toronto --scan data/Toronto_3D/L004.ply \
  --onnx outputs/toronto_onnx/pointnet_toronto_w8a8_perchannel.onnx
```

Generate Toronto-calibrated ONNX Runtime QDQ variants. Calibration scans should be representative and should not include the held-out evaluation scan:

```bash
python -m src.quantize_toronto --calib-scans L001.ply L002.ply
```

Artifacts are isolated under `outputs/toronto_onnx/`. [notebooks/04_toronto_quantization_deep_dive.ipynb](notebooks/04_toronto_quantization_deep_dive.ipynb) compares PyTorch and ONNX inference, evaluates PTQ variants on held-out tiles, and inspects QDQ scales and zero-points. Toronto preprocessing uses label-free median-height centering so the same feature transform works for training, calibration, and deployment.

## Notebooks

* **`01_point_cloud_data.ipynb`**: what scans look like (3D, top-down, side views), class balance, density and
  noise vs range, intensity, the model's input tensor, permutation invariance, voxel downsampling,
  range-image projection, and predictions of the trained model with a confusion matrix.
* **`02_quantization_deep_dive.ipynb`**: scale/zero-point by hand, symmetric vs asymmetric, why to normalize
  inputs, BatchNorm folding, per-tensor vs per-channel weights, activation histograms and the clipping/rounding
  trade-off, reading a QDQ ONNX graph, accuracy/latency of every variant, and a layer-by-layer SQNR plot.
* **`03_toronto_3d_eda.ipynb`**: load Toronto_3D scans and perform exploratory data analysis.
* **`04_toronto_quantization_deep_dive.ipynb`**: Toronto_3D baseline inference, PTQ calibration, held-out accuracy/performance comparison, and QDQ inspection.

### Range-image projection

A range image turns an unordered 3D scan into a 2D grid indexed by the direction each laser return came from.
First, coordinates are made relative to the scanner (in this project, the scanner is at `(0, 0, SCANNER_HEIGHT)`).
For a relative point `(x, y, z)`, its **range** is the distance $r = \sqrt{x^2+y^2+z^2}$, its **yaw** is the
horizontal angle around the vertical axis, `atan2(y, x)`, and its **pitch** is the vertical angle
above or below the horizontal plane, $\arcsin(z/r)$. Both angles are in radians in the code. Yaw selects the
horizontal image coordinate; pitch selects the vertical row. The configured vertical field of view (here, from
$-35^\circ$ to $25^\circ$) is divided into `H` rows, while yaw covers the full 360° in `W` columns.

Each pixel stores the range of the return in that direction; a second image can store its class label. Directions
with no return remain empty. If multiple points land in one pixel, this example keeps the nearest return, like a
depth/range camera. The result is a regular 2D tensor that a conventional image model can process, though the
projection can discard points when several returns collide in the same pixel.

## Project layout

```
src/data.py                 synthetic scan generator + Dataset (6 classes)
src/model.py                PointNet segmentation (Conv1d/BN/ReLU, max-pool, concat; no kNN)
src/train.py                AdamW + OneCycle, class-weighted CE, best-by-val-mIoU checkpoint
src/export_onnx.py          torch.onnx export (opset 18, static shapes) + ORT parity check
src/quantize.py             ONNX Runtime static PTQ, QDQ format, 6 variants
src/export_toronto_onnx.py  Toronto_3D checkpoint export and optional parity check
src/infer_toronto.py        tiled Toronto inference with per-point PLY predictions
src/quantize_toronto.py     Toronto-calibrated ONNX Runtime static PTQ
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
