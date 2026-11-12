#!/usr/bin/env bash
# scrollprize/ink_9um maps (both released seeds, both depth directions) of surface-volume zarrs:
#   run_ink9um.sh <out dir> <zarr path or URL> [<zarr> ...]
# A URL of a published segment is named after its segment id, a local <name>.zarr after <name>.
# Writes <name>_ink9um_s<seed>.tif (layers as stored) and <name>_ink9um_s<seed>_reverse.tif.
# Needs villa's vesuvius package at 5a4388f (PYTHONPATH below; it has no C++ dependency there),
# $POSTEL/ink/.venv and the two step-075000 checkpoints in $POSTEL/ink/models/ink_9um.
set -uo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
export PYTHONPATH="$POSTEL/villa/vesuvius/src"
out="$1"; shift
mkdir -p "$out"
for src in "$@"; do
  case "$src" in
    http*/segments/*) name="${src#*/segments/}"; name="${name%%/*}" ;;
    *) name="$(basename "${src%/}" .zarr)" ;;
  esac
  for seed in 42 43; do
    tif="$out/${name}_ink9um_s$seed.tif"
    [ -f "$tif" ] && continue
    "$POSTEL/ink/.venv/bin/python" -m vesuvius.ink_detection.inference.infer "$src" \
      "$POSTEL/ink/models/ink_9um/hybrid_3d2d-seed$seed/step-075000.pth" "$tif" \
      --overlap 0.5 --blend-mode hann --batch-size "${BATCH:-16}" --direction both --no-compile \
      > "$out/${name}_ink9um_s$seed.log" 2>&1 || echo "FAILED $name seed $seed"
  done
  echo "$(date '+%T') $name done"
done
