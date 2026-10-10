"""Optional: the same export -> quantize -> compare loop on a *downloaded, pretrained* model.

    python -m src.pretrained_image_seg                 # downloads ~13 MB of weights on first run
    python -m src.pretrained_image_seg --image my.jpg  # use your own photo

Model: torchvision LR-ASPP with a MobileNetV3-Large backbone, pretrained for semantic
segmentation on a COCO subset with the 21 Pascal VOC classes. MobileNetV3 is a classic edge
architecture (depthwise separable convs, hard-swish, squeeze-and-excitation), so it shows
quantization effects that the small PointNet does not:
* depthwise convs have few weights per channel -> per-channel quantization matters a lot,
* hard-swish / SE blocks create activation ranges that are harder to calibrate.

Outputs land in outputs/pretrained/: fp32 and int8 ONNX files plus a side-by-side PNG of
the float and int8 segmentation masks and their pixel agreement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

OUT = Path("outputs/pretrained")
VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class LogitsOnly(torch.nn.Module):
    """torchvision segmentation models return a dict; ONNX export wants a tensor."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)["out"]


def load_image(path: str | None, size: int) -> np.ndarray:
    from PIL import Image

    if path:
        img = Image.open(path).convert("RGB")
    else:  # no image given: a random test pattern, enough to exercise the pipeline
        rng = np.random.default_rng(0)
        img = Image.fromarray((rng.random((size, size, 3)) * 255).astype(np.uint8))
        print("No --image given: using a random test pattern. Pass --image <photo.jpg> for meaningful masks.")
    img = img.resize((size, size))
    return np.asarray(img).astype(np.float32) / 255.0


def to_input(img: np.ndarray) -> np.ndarray:
    return ((img - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--no-weights", action="store_true", help="skip the download (random weights; for testing the pipeline)")
    args = ap.parse_args()

    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights, lraspp_mobilenet_v3_large

    from .evaluate import make_session

    OUT.mkdir(parents=True, exist_ok=True)
    weights = None if args.no_weights else LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    net = lraspp_mobilenet_v3_large(weights=weights, weights_backbone=None).eval()
    model = LogitsOnly(net)

    img = load_image(args.image, args.size)
    x = to_input(img)

    fp32 = OUT / "lraspp_mbv3_fp32.onnx"
    torch.onnx.export(model, (torch.from_numpy(x),), str(fp32), input_names=["image"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    prep = OUT / "lraspp_mbv3_fp32_prep.onnx"
    quant_pre_process(str(fp32), str(prep), skip_symbolic_shape=True)

    class Reader(CalibrationDataReader):
        """Calibration from flipped / brightness-jittered copies of the image. With real data,
        use a few hundred representative images instead."""

        def __init__(self):
            rng = np.random.default_rng(0)
            self.samples = []
            for _ in range(16):
                aug = img[:, ::-1] if rng.random() < 0.5 else img
                aug = np.clip(aug * rng.uniform(0.7, 1.3), 0, 1)
                self.samples.append(to_input(aug))
            self.i = 0

        def get_next(self):
            if self.i >= len(self.samples):
                return None
            self.i += 1
            return {"image": self.samples[self.i - 1]}

    int8 = OUT / "lraspp_mbv3_w8a8.onnx"
    quantize_static(str(prep), str(int8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)

    f_logits = make_session(str(prep)).run(None, {"image": x})[0]
    q_logits = make_session(str(int8)).run(None, {"image": x})[0]
    f_mask, q_mask = f_logits.argmax(1)[0], q_logits.argmax(1)[0]
    agree = float((f_mask == q_mask).mean())
    from .metrics import sqnr_db

    print(f"fp32 {fp32.stat().st_size/1e6:.1f} MB  ->  int8 {int8.stat().st_size/1e6:.1f} MB")
    print(f"logit SQNR {sqnr_db(f_logits, q_logits):.1f} dB, pixel agreement fp32 vs int8: {agree*100:.1f}%")
    classes = [VOC_CLASSES[c] for c in np.unique(f_mask)]
    print("classes predicted (fp32):", classes)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img); axs[0].set_title("input")
    axs[1].imshow(f_mask, cmap="tab20", vmin=0, vmax=20); axs[1].set_title("fp32 mask")
    axs[2].imshow(q_mask, cmap="tab20", vmin=0, vmax=20); axs[2].set_title(f"int8 mask ({agree*100:.1f}% agree)")
    for a in axs:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fp32_vs_int8.png", dpi=120)
    print(f"-> {OUT / 'fp32_vs_int8.png'}")


if __name__ == "__main__":
    main()
