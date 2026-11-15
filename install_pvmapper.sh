#!/usr/bin/env bash
# Sets up pvmapper in a virtualenv on Ubuntu/Debian.
#   ./install_pvmapper.sh rocm   # AMD Radeon (default)
#   ./install_pvmapper.sh cuda   # NVIDIA
#   ./install_pvmapper.sh cpu
set -euo pipefail

BACKEND="${1:-rocm}"
ROCM_INDEX="${ROCM_INDEX:-https://download.pytorch.org/whl/rocm7.2}"
cd "$(dirname "$0")"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

case "$BACKEND" in
  rocm) pip install torch torchvision --index-url "$ROCM_INDEX" ;;
  cuda) pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 ;;
  cpu)  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu ;;
  *) echo "unknown backend $BACKEND (rocm|cuda|cpu)"; exit 1 ;;
esac

pip install -r requirements-pvmapper.txt pyogrio

python -m pvmapper device
python -m pvmapper models          # downloads the default model weights (~630 MB)

cat <<EOF

Fertig. Start der Oberfläche:
  source .venv/bin/activate && python -m pvmapper web
  → http://127.0.0.1:8765

Wird die Radeon nicht erkannt ("device": "cpu"), vor dem Start versuchen:
  export HSA_OVERRIDE_GFX_VERSION=11.0.0
EOF
