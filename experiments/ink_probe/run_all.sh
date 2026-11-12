#!/usr/bin/env bash
# Every ink map of the probe, one model after another:
#   [HECATE_BATCH=n] [V8_BATCH=n] run_all.sh    (log: $POSTEL/logs/ink_all.log, ends with INKALLDONE)
# Hecate on the labelled PHerc. 1447 stacks in segments/PHerc1447_lab (winding 58 here) and v8-in
# on a labelled crop of it (a check of this installation against the dataset's own v8-in map, and
# the wrong depth direction as a control), then Hecate, ink_9um and v8-in on the six published
# PHerc. 0800 segments. Every step skips maps that already exist.
# On the 12 GB card: Hecate at batch 16 took 9 GB, and v8-in ran out of memory at batch 12 and took
# 7 GB at batch 6, at about 19 tiles per second. The 0800 pass therefore uses a tile stride of 32 px
# instead of the model's 21. Smaller batches leave room for a spiral fit beside a model.
set -uo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
E="$(cd "$(dirname "$0")" && pwd)"
I="$POSTEL/ink"
{
  bash "$E/run_hecate.sh" "$I/segments/PHerc1447_lab" "$I/out/PHerc1447" "${HECATE_BATCH:-16}"
  BATCH="${V8_BATCH:-6}" bash "$E/run_v8in.sh" "$I/out/PHerc1447" "$I/segments/PHerc1447_crop/w058c_layers"
  bash "$E/run_hecate.sh" "$I/segments/PHerc0800" "$I/out/PHerc0800" "${HECATE_BATCH:-16}"
  bash "$E/run_ink9um.sh" "$I/out/PHerc0800" "$I"/segments/PHerc0800/*.zarr
  BATCH="${V8_BATCH:-6}" STRIDE=32 bash "$E/run_v8in.sh" "$I/out/PHerc0800" "$I"/segments/PHerc0800/*_layers
  echo INKALLDONE
} > "$POSTEL/logs/ink_all.log" 2>&1
