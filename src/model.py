"""PointNet-style semantic segmentation network, kept deployment-friendly.

PointNet (Qi et al., CVPR 2017) processes every point with the same small MLP, written here
as 1x1 Conv1d layers, then max-pools over all points to get one global scene feature.
Concatenating each point's local feature with the global feature lets the head classify
points using both local shape and scene context.

Why this variant is "edge friendly":
* only Conv (1x1), BatchNorm (folds into Conv at export), ReLU, ReduceMax, Expand, Concat,
* a fixed number of points N, so every tensor shape is static,
* no k-NN, gather or scatter ops, which many NPUs do not support well.

The original paper also has small "T-Net" alignment networks; they are left out because
they add little for gravity-aligned scans and complicate quantization.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, c_in: int, c_out: int):
        super().__init__(
            nn.Conv1d(c_in, c_out, kernel_size=1, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
        )


class PointNetSeg(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 6, local_dim: int = 64, global_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.local = nn.Sequential(ConvBNReLU(in_channels, 64), ConvBNReLU(64, local_dim))
        self.glob = nn.Sequential(ConvBNReLU(local_dim, 128), ConvBNReLU(128, global_dim))
        self.head = nn.Sequential(
            ConvBNReLU(local_dim + global_dim, 256),
            ConvBNReLU(256, 128),
            nn.Dropout(dropout),
            nn.Conv1d(128, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, N) -> logits (B, num_classes, N)."""
        n = x.shape[2]
        local = self.local(x)                          # (B, 64, N)  per-point features
        g = self.glob(local)                           # (B, 256, N)
        g = torch.amax(g, dim=2, keepdim=True)         # (B, 256, 1) order-invariant scene feature
        g = g.expand(-1, -1, n)                        # (B, 256, N)
        return self.head(torch.cat([local, g], dim=1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_macs(model: PointNetSeg, n_points: int) -> int:
    """Multiply-accumulates per scene: every 1x1 conv costs C_in * C_out per point."""
    return sum(m.in_channels * m.out_channels * n_points for m in model.modules() if isinstance(m, nn.Conv1d))


def pick_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path: str, device: str | torch.device = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = PointNetSeg(**ckpt["model_args"])
    if ckpt.get("qat"):
        from .qat import prepare_qat_model

        model = prepare_qat_model(model)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
