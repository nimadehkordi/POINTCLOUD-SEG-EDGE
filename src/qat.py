"""Fake-quantization helpers for W8A8 quantization-aware training."""

from __future__ import annotations

import torch
import torch.ao.quantization as tq
import torch.nn as nn
from torch.ao.quantization import FakeQuantize
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver


def qat_qconfig() -> tq.QConfig:
    """Unsigned per-tensor activation and symmetric per-output-channel weight QAT."""
    return tq.QConfig(
        activation=FakeQuantize.with_args(
            observer=MinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
        ),
        weight=FakeQuantize.with_args(
            observer=PerChannelMinMaxObserver,
            quant_min=-127,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        ),
    )


def prepare_qat_model(model: torch.nn.Module) -> torch.nn.Module:
    """Fuse Conv-BN-ReLU blocks and insert fake quantization for QAT."""
    fusion_groups = [
        [f"{block}.{layer}" for layer in ("0", "1", "2")]
        for block in ("local.0", "local.1", "glob.0", "glob.1", "head.0", "head.1")
    ]
    tq.fuse_modules_qat(model, fusion_groups, inplace=True)
    model.train()
    model.qconfig = qat_qconfig()
    return tq.prepare_qat(model, inplace=False)


def to_float_export_model(qat_model: nn.Module) -> nn.Module:
    """Bake QAT weight fake-quant and folded BatchNorm into an ordinary float model.

    Activation fake quantization trains robustness to quantization ranges, but is omitted
    from this float ONNX intermediate; run the ONNX Runtime PTQ step on it to create QDQ.
    """
    from .model import PointNetSeg

    source = qat_model.cpu().eval()
    float_model = PointNetSeg(
        in_channels=source.local[0][0].in_channels,
        num_classes=source.head[3].out_channels,
        local_dim=source.local[1][0].out_channels,
        global_dim=source.glob[1][0].out_channels,
        dropout=source.head[2].p,
    ).cpu().eval()
    pairs = (
        (source.local[0][0], float_model.local[0][0]),
        (source.local[1][0], float_model.local[1][0]),
        (source.glob[0][0], float_model.glob[0][0]),
        (source.glob[1][0], float_model.glob[1][0]),
        (source.head[0][0], float_model.head[0][0]),
        (source.head[1][0], float_model.head[1][0]),
    )
    with torch.no_grad():
        for qat, conv in pairs:
            weight = qat.weight_fake_quant(qat.weight) if hasattr(qat, "weight_fake_quant") else qat.weight
            bn = qat.bn
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            conv.weight.copy_(weight * scale.reshape(-1, 1, 1))
            bias = qat.bias if qat.bias is not None else torch.zeros_like(bn.running_mean)
            conv.bias = nn.Parameter(bn.bias + (bias - bn.running_mean) * scale)
        final = source.head[3]
        fake_quant = getattr(final, "weight_fake_quant", None)
        final_weight = fake_quant(final.weight) if fake_quant is not None else final.weight
        float_model.head[3].weight.copy_(final_weight)
        if final.bias is not None:
            float_model.head[3].bias.copy_(final.bias)
    return float_model