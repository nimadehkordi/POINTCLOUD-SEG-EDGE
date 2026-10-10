#!/usr/bin/env bash
# Full pipeline: train -> evaluate -> export ONNX -> quantize -> compare layers.
# Quick version for a first run:  QUICK=1 ./run_all.sh   (about 2-3 minutes)
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate

if [ "${QUICK:-0}" = "1" ]; then
  TRAIN_ARGS="--epochs 4 --train-size 400 --val-size 100"
else
  TRAIN_ARGS=""
fi

echo "== 1/5 train =="
python -m src.train $TRAIN_ARGS
echo "== 2/5 export ONNX =="
python -m src.export_onnx
echo "== 3/5 quantize =="
python -m src.quantize
echo "== 4/5 evaluate accuracy + latency =="
python -m src.evaluate
echo "== 5/5 layer-by-layer comparison =="
python -m src.compare_layers --variants w8a8_pertensor w8a8_perchannel w8a16

echo
echo "All done. Open the notebooks:  jupyter lab notebooks/"
