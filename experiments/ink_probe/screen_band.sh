#!/usr/bin/env bash
# Read every winding of a fitted band with one ink model.
#   screen_band.sh <fit's winding meshes> <work folder> <scan level-0 URL> <umbilicus.json> \
#       <surface prediction zarr URL> <first winding> <last winding>
# <fit's winding meshes> is <run>/meshes/fitted as fit_spiral.py writes it (w###/ tifxyz folders;
# run the fit with FIT_SPIRAL_SKIP_SAVE_MESH unset). Per winding, in batches of $BATCH (16):
#   1. snap: the PHerc. 343 First Letters submission's snap_meshes.py moves every node along its
#      normal onto the nearest sheet of the surface prediction (github.com/Nieuwlaar/
#      pherc343-first-letters, in $POSTEL/data/PHerc0343/winner; needs $POSTEL/snapenv with zarr 2);
#      its report keeps, per winding, the share of nodes that were within 2 voxels of a sheet;
#   2. render: 17 layers from -6.5 voxels, the window the 9 um models read at depth +1, with the
#      normals turned away from the scroll's axis (render_mesh.py, $JOBS windings at a time);
#   3. ink: villa's inference on the layers in reversed order, the side that faces the axis,
#      where a rolled scroll carries its text; model $CKPT (default: the submission's fine-tune);
#   4. the rendered layers are deleted, the maps kept: <work>/seg/w###/preds/*.tif.
# A winding whose map exists is skipped, so the script can be run again after a stop.
set -uo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
meshes="$1"; work="$2"; scan="$3"; umbilicus="$4"; surface="$5"; first="$6"; last="$7"
E="$POSTEL/windcon/experiments/ink_probe"
CKPT="${CKPT:-$POSTEL/ink/ckpt343/dense_native-016000.rebuilt.pth}"
BATCH="${BATCH:-16}"; JOBS="${JOBS:-6}"
py="$POSTEL/windcon/.venv/bin/python"
export PYTHONPATH="$POSTEL/villa/vesuvius/src" TMPDIR="$POSTEL/t"
mkdir -p "$work/snapped" "$work/seg" "$TMPDIR"
log() { echo "$(date '+%T') $*"; }

todo=()
for i in $(seq "$first" "$last"); do
  w=$(printf 'w%03d' "$i")
  [ -f "$meshes/$w/x.tif" ] || continue
  ls "$work/seg/$w/preds"/*.tif > /dev/null 2>&1 || todo+=("$w")
done
log "${#todo[@]} windings to read"

for ((start = 0; start < ${#todo[@]}; start += BATCH)); do
  batch=("${todo[@]:start:BATCH}")
  log "batch ${batch[0]}..${batch[-1]}"
  need=()
  for w in "${batch[@]}"; do [ -f "$work/snapped/$w/x.tif" ] || need+=("$w"); done
  if [ "${#need[@]}" -gt 0 ]; then
    printf '%s\n' "${need[@]}" > "$work/snap_list.txt"
    SURF_ZARR="$surface" "$POSTEL/snapenv/bin/python" "$POSTEL/data/PHerc0343/winner/mesh/snap_meshes.py" \
      "$meshes" band "$work/snapped" --windings "$work/snap_list.txt" --procs 8 >> "$work/snap.log" 2>&1 \
      || log "snap FAILED (see $work/snap.log)"
  fi
  printf '%s\n' "${batch[@]}" | xargs -P "$JOBS" -I{} sh -c \
    "[ -f '$work/snapped/{}/x.tif' ] && '$py' '$E/render_mesh.py' '$work/snapped/{}' --volume '$scan' --umbilicus '$umbilicus' \
       --layers 17 --first-offset -6.5 --zarr '$work/seg/{}/{}.zarr' > '$work/seg/render_{}.log' 2>&1"
  "$POSTEL/ink/.venv/bin/python" -m vesuvius.ink_detection.inference.infer --folder "$work/seg" \
    --checkpoint-path "$CKPT" --direction reverse --overlap 0.5 --blend-mode hann \
    --batch-size "${INK_BATCH:-8}" --no-compile >> "$work/ink.log" 2>&1 || log "ink FAILED (see $work/ink.log)"
  for w in "${batch[@]}"; do
    ls "$work/seg/$w/preds"/*.tif > /dev/null 2>&1 && rm -rf "$work/seg/$w/$w.zarr"
  done
  log "maps so far: $(ls "$work"/seg/*/preds/*.tif 2> /dev/null | wc -l)"
done
log "done"
