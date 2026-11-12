#!/usr/bin/env bash
# Hecate 9.6 um ink maps for every stack resampled to 9.6 um in a folder, in both depth directions
# (the orientation of a published render is unknown):  run_hecate.sh <stacks dir> <out dir> [batch]
# Needs $POSTEL/ink/.venv (torch, numpy, zarr, numcodecs) and scrollprize/hecate in $POSTEL/ink/models.
set -uo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
stacks="$1"; out="$2"; batch="${3:-16}"
H="$POSTEL/ink/models/hecate"
export TMPDIR="$POSTEL/scratch"
mkdir -p "$out"
for f in "$stacks"/*_9.6um.npy; do
  name="$(basename "$f" _9.6um.npy)"
  for dir in fwd rev; do
    png="$out/${name}_hecate_$dir.png"
    [ -f "$png" ] && continue
    flag=""; [ "$dir" = rev ] && flag="--reverse"
    "$POSTEL/ink/.venv/bin/python" "$H/hecate.py" --checkpoint "$H/hecate_9.6um.pth" --input "$f" \
      --spacing-um 9.6 --output "$png" --device cuda --precision bf16 --batch-size "$batch" $flag \
      || echo "FAILED $name $dir"
  done
  echo "$(date '+%T') $name done"
done
