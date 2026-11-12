#!/usr/bin/env bash
# v8-in ink maps (YoussefMoNader/ink-8um-v8in, the 8 um base model that found the PHerc. 1447
# letters) of layer-TIFF folders, in both depth directions:
#   [BATCH=n] [STRIDE=px] run_v8in.sh <out dir> <layers dir> [<layers dir> ...]
# A folder <name>_layers or <name>/layers is named <name>. Writes <name>_v8in_fwd.png (layers as
# stored) and <name>_v8in_rev.png. Needs $POSTEL/ink/.venv and the model in $POSTEL/ink/models/v8in.
# The default batch of 6 takes 7 GB (12 ran out of memory on a 12 GB card).
set -uo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
out="$1"; shift
mkdir -p "$out"
for layers in "$@"; do
  name="$(basename "${layers%/}")"
  [ "$name" = layers ] && name="$(basename "$(dirname "${layers%/}")")"
  name="${name%_layers}"
  for dir in fwd rev; do
    png="$out/${name}_v8in_$dir.png"
    [ -f "$png" ] && continue
    flag="--no-reverse"; [ "$dir" = rev ] && flag="--reverse"
    "$POSTEL/ink/.venv/bin/python" "$POSTEL/ink/models/v8in/predict.py" --layers "$layers" --output "$png" \
      $flag --batch-size "${BATCH:-6}" ${STRIDE:+--stride "$STRIDE"} > "$out/${name}_v8in_$dir.log" 2>&1 \
      || echo "FAILED $name $dir"
  done
  echo "$(date '+%T') $name done"
done
