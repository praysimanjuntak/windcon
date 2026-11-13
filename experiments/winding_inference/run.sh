#!/usr/bin/env bash
# Run a script of this folder in villa's spiral-fitting environment, with the vc stand-in on the path:
#   run.sh build_store.py --checkpoint <run>/checkpoint_fitted.ckpt --umbilicus <umbilicus.json> \
#       --volume <chunk cache folder> --model <ckpt_final.pth> --out <store folder> [...]
#   run.sh transfer_check.py --fine <zarr URL> --coarse <zarr URL> --model <ckpt_final.pth> --out <name>.npz
# The model is `hf download scrollprize/winding_model_9um`. A chunk cache folder is the one under
# windcon's data/cache whose SOURCE file names the volume level the fit was made in.
# The scan is interpolated on the GPU; VC_SHIM_DEVICE=cpu does it on the CPU (four times the CPU time,
# but faster when another job shares the GPU).
set -euo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
here=$(cd "$(dirname "$0")" && pwd)
script="$1"; shift
cd "$POSTEL/villa/spiral-fitting"
export SPIRAL_SCRIPTS_DIR="$PWD"
export VC_SHIM_DEVICE="${VC_SHIM_DEVICE:-cuda}"
export PYTHONPATH="$POSTEL/villa/vesuvius/src:$POSTEL/windcon/src:$here/vc_shim${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python "$here/$script" "$@"
