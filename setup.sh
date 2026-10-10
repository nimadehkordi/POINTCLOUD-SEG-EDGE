#!/usr/bin/env bash
# Create a virtual environment and install everything. Run once:  ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
if ! command -v "$PY" >/dev/null; then
  echo "python3 not found. On macOS: brew install python@3.12" >&2
  exit 1
fi

"$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name pointcloud-seg-edge --display-name "Python (pointcloud-seg-edge)"

python - <<'EOF'
import torch, onnx, onnxruntime
mps = torch.backends.mps.is_available()
print(f"torch {torch.__version__} (MPS GPU: {'yes' if mps else 'no'}), onnx {onnx.__version__}, onnxruntime {onnxruntime.__version__}")
EOF
echo
echo "Done. Next:  source .venv/bin/activate && ./run_all.sh"
