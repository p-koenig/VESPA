#!/usr/bin/env bash
# Create the separate environment of the upstream SYMPOL code and a patched checkout of it.
#   experiments/pretraining/mountaincar/setup_sympol.sh
# third_party/sympol      upstream checkout at commit 4f24add with the export hook applied
# third_party/sympol-env  conda prefix with Python 3.11.4 and the pinned packages of environment/sympol_paper
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
DIR=$REPO/third_party/sympol
ENVP=$REPO/third_party/sympol-env
if [ ! -d "$DIR/.git" ]; then
  git clone https://github.com/s-marton/SYMPOL "$DIR"
  git -C "$DIR" checkout 4f24add
  git -C "$DIR" apply "$HERE/sympol_export_hook.patch"
fi
if [ ! -x "$ENVP/bin/python" ]; then
  conda create -y -p "$ENVP" -c conda-forge --override-channels python=3.11.4 pip
fi
"$ENVP/bin/python" -m pip install -r "$REPO/environment/sympol_paper/requirements.txt" \
  -c "$REPO/environment/sympol_paper/pip_freeze.txt"
JAX_PLATFORMS=cpu "$ENVP/bin/python" -c \
  "import jax, flax, optax, gymnax; print('SYMPOL environment OK', jax.__version__, flax.__version__)"
