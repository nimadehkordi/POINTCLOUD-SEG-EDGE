"""Layer-by-layer comparison of the float model against quantized models.

    python -m src.compare_layers                                  # all quantized variants
    python -m src.compare_layers --variants w8a8_pertensor w8a16

This is the same idea as comparing a float reference with the quantized device build:
expose every intermediate tensor, run the same input through both models, and measure
how much each layer deviates. Where the SQNR drops sharply is where quantization hurts.

How the matching works: the quantizer wraps a float tensor T as
    T -> QuantizeLinear -> DequantizeLinear -> (consumers)
so the dequantized output is "T as the integer pipeline sees it". We compare it with T in
the float model. Both sessions run with graph optimizations disabled, so ORT computes the
QDQ graph exactly as written (simulated quantization) instead of fusing it into integer
kernels; the numbers are therefore deterministic and easy to reason about.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from .data import SyntheticScans
from .evaluate import make_session
from .metrics import cosine_similarity, sqnr_db

ONNX_DIR = Path("outputs/onnx")
FP32_PREP = ONNX_DIR / "pointnet_seg_fp32_prep.onnx"


def float_tensors_in_order(model: onnx.ModelProto) -> list[tuple[str, str, str]]:
    """(tensor_name, producing_node_name, op_type) for every node output, in graph order."""
    out = []
    for node in model.graph.node:
        for t in node.output:
            out.append((t, node.name or t, node.op_type))
    return out


def qdq_mapping(qmodel: onnx.ModelProto) -> dict[str, str]:
    """float tensor name -> name of its DequantizeLinear(QuantizeLinear(.)) output in the QDQ model."""
    q_out_to_src = {n.output[0]: n.input[0] for n in qmodel.graph.node if n.op_type == "QuantizeLinear"}
    mapping = {}
    for n in qmodel.graph.node:
        if n.op_type == "DequantizeLinear" and n.input[0] in q_out_to_src:
            mapping[q_out_to_src[n.input[0]]] = n.output[0]
    return mapping


def with_extra_outputs(model: onnx.ModelProto, names: list[str]) -> onnx.ModelProto:
    m = onnx.ModelProto()
    m.CopyFrom(model)
    existing = {o.name for o in m.graph.output}
    for name in names:
        if name not in existing:
            m.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    return m


def compare(fp32_path: Path, quant_path: Path, n_scenes: int = 8, n_points: int = 2048) -> list[dict]:
    fmodel, qmodel = onnx.load(str(fp32_path)), onnx.load(str(quant_path))
    mapping = qdq_mapping(qmodel)
    order = [(t, node, op) for t, node, op in float_tensors_in_order(fmodel) if t in mapping]
    # The final logits are produced in float by the last DequantizeLinear (or a float head); add them too.
    final = fmodel.graph.output[0].name
    if final not in mapping:
        order.append((final, "output", "logits"))
        mapping[final] = final

    ftmp, qtmp = fp32_path.with_suffix(".debug.onnx"), quant_path.with_suffix(".debug.onnx")
    onnx.save(with_extra_outputs(fmodel, [t for t, _, _ in order]), str(ftmp))
    onnx.save(with_extra_outputs(qmodel, [mapping[t] for t, _, _ in order]), str(qtmp))
    fs, qs = make_session(str(ftmp), optimize=False), make_session(str(qtmp), optimize=False)
    fnames = [o.name for o in fs.get_outputs()]
    qnames = [o.name for o in qs.get_outputs()]

    ds = SyntheticScans("test", n_scenes, n_points)
    acc = {t: {"sqnr": [], "cos": [], "maxerr": [], "range": []} for t, _, _ in order}
    for i in range(n_scenes):
        feed = {"points": ds[i][0][None].numpy()}
        fo = dict(zip(fnames, fs.run(None, feed)))
        qo = dict(zip(qnames, qs.run(None, feed)))
        for t, _, _ in order:
            a, b = fo[t], qo[mapping[t]]
            acc[t]["sqnr"].append(sqnr_db(a, b))
            acc[t]["cos"].append(cosine_similarity(a, b))
            acc[t]["maxerr"].append(float(np.abs(a - b).max()))
            acc[t]["range"].append(float(np.abs(a).max()))
    ftmp.unlink()
    qtmp.unlink()

    rows = []
    for idx, (t, node, op) in enumerate(order):
        a = acc[t]
        rows.append({"idx": idx, "tensor": t, "node": node, "op": op,
                     "sqnr_db": float(np.mean(a["sqnr"])), "cosine": float(np.mean(a["cos"])),
                     "max_abs_err": float(np.max(a["maxerr"])), "max_abs_value": float(np.max(a["range"]))})
    return rows


def plot(results: dict[str, list[dict]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, rows in results.items():
        ax.plot([r["sqnr_db"] for r in rows], marker="o", ms=3, label=name)
    first = next(iter(results.values()))
    ax.set_xticks(range(len(first)))
    ax.set_xticklabels([f"{r['op']}:{r['tensor'][:22]}" for r in first], rotation=75, fontsize=7)
    ax.set_ylabel("SQNR vs float (dB, higher is better)")
    ax.set_title("Where does quantization error enter? Layer-by-layer SQNR")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--n-scenes", type=int, default=8)
    args = ap.parse_args()

    paths = sorted(p for p in ONNX_DIR.glob("pointnet_seg_w*.onnx") if not p.stem.endswith(".debug"))
    if args.variants:
        paths = [p for p in paths if p.stem.replace("pointnet_seg_", "") in args.variants]
    results = {}
    out_dir = Path("outputs/layer_compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        name = p.stem.replace("pointnet_seg_", "")
        rows = compare(FP32_PREP, p, args.n_scenes)
        results[name] = rows
        with open(out_dir / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n== {name} ==")
        print(f"{'#':>3s} {'op':10s} {'tensor':40s} {'SQNR dB':>8s} {'cosine':>8s} {'max|x|':>8s}")
        for r in rows:
            print(f"{r['idx']:3d} {r['op']:10s} {r['tensor'][:40]:40s} {r['sqnr_db']:8.1f} {r['cosine']:8.5f} {r['max_abs_value']:8.2f}")
    if results:
        plot(results, out_dir / "layer_sqnr.png")
        print(f"\nplot -> {out_dir / 'layer_sqnr.png'}")


if __name__ == "__main__":
    main()
