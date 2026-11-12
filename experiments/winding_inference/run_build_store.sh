#!/usr/bin/env bash
# Run build_store.py in villa's spiral-fitting environment with the vc stand-in on the path.
#   run_build_store.sh --checkpoint <run>/checkpoint_fitted.ckpt --umbilicus <umbilicus.json> \
#       --volume <chunk cache folder> --model <ckpt_final.pth> --out <store folder> [...]
# The model is `hf download scrollprize/winding_model_9um`. The chunk cache folder is the one under
# windcon's data/cache whose SOURCE file names the volume level the fit was made in.
set -euo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
here=$(cd "$(dirname "$0")" && pwd)
cd "$POSTEL/villa/spiral-fitting"
export SPIRAL_SCRIPTS_DIR="$PWD"
export PYTHONPATH="$POSTEL/villa/vesuvius/src:$POSTEL/windcon/src:$here/vc_shim${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python "$here/build_store.py" "$@"
