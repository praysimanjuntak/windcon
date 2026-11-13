#!/usr/bin/env bash
# Read every winding of a fitted band on both faces and lay the maps out for triage_lane.sh.
#   screen_band.sh <fit's winding meshes> <lane folder> <scan level-0 URL> <umbilicus.json> \
#       <surface prediction zarr URL> <voxel size in um> <first winding> <last winding>
# <fit's winding meshes> is <run>/meshes/fitted as fit_spiral.py writes it (w###/ tifxyz folders;
# run the fit with FIT_SPIRAL_SKIP_SAVE_MESH unset). The steps are those the PHerc. 343 control
# passed (fl343_own_fit.sh), on whole windings:
#   1. snap: the PHerc. 343 First Letters submission's snap_meshes.py moves every node along its
#      normal onto the nearest sheet of the surface prediction (github.com/Nieuwlaar/
#      pherc343-first-letters, in $POSTEL/data/PHerc0343/winner; needs $POSTEL/snapenv with zarr 2);
#      its report keeps, per winding, the share of nodes that were within 2 voxels of a sheet;
#   2. render: 66 layers round the snapped mesh, the normals turned away from the scroll's axis
#      (render_mesh.py, fetch_layers.py);
#   3. ink: the models in $MODELS (default: the submission's three villa models) at depth +1, the
#      layers as stored and reversed (run_fl343_models.sh). Reversed is the face towards the axis;
#   4. layout: lane_layout.py writes <lane>/maps/w### and <lane>/renders/w###.zarr; the render is
#      deleted, the ink maps stay in <lane>/ink/w###.
# A winding whose layout exists is skipped, so the script can be run again after a stop.
set -uo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
meshes="$1"; lane="$2"; scan="$3"; umbilicus="$4"; surface="$5"; um="$6"; first="$7"; last="$8"
E="$POSTEL/windcon/experiments/ink_probe"
py="$POSTEL/windcon/.venv/bin/python"
read -r -a models <<< "${MODELS:-dnative hybrid_3d2d-seed42 hybrid_3d2d-seed43}"
mkdir -p "$lane/snapped" "$lane/work" "$lane/ink"
lane="$(cd "$lane" && pwd)"
log() { echo "$(date '+%T') $*"; }

todo=(); need=()
for i in $(seq "$first" "$last"); do
  w=$(printf 'w%03d' "$i")
  [ -f "$meshes/$w/x.tif" ] || continue
  [ -f "$lane/maps/$w/done_d0" ] && continue
  todo+=("$w")
  [ -f "$lane/snapped/$w/x.tif" ] || need+=("$w")
done
log "${#todo[@]} windings to read, ${#need[@]} to snap"
if [ "${#need[@]}" -gt 0 ]; then
  printf '%s\n' "${need[@]}" > "$lane/snap_list.txt"
  SURF_ZARR="$surface" "$POSTEL/snapenv/bin/python" "$POSTEL/data/PHerc0343/winner/mesh/snap_meshes.py" \
    "$meshes" band "$lane/snapped" --windings "$lane/snap_list.txt" --procs 8 >> "$lane/snap.log" 2>&1 \
    || log "snap FAILED (see $lane/snap.log)"
fi

cd "$POSTEL/windcon"
for w in "${todo[@]}"; do
  [ -f "$lane/snapped/$w/x.tif" ] || { log "$w: no snapped mesh"; continue; }
  "$py" "$E/render_mesh.py" "$lane/snapped/$w" --volume "$scan" --umbilicus "$umbilicus" --layers 66 \
    --voxel-um "$um" --out "$lane/work/${w}_layers" > "$lane/work/render_$w.log" 2>&1 \
    || { log "$w: render FAILED (see $lane/work/render_$w.log)"; continue; }
  "$py" "$E/fetch_layers.py" "$lane/work/${w}_layers" --name "$w" --native-spacing "$um" --out "$lane/work" --zarr \
    | tail -n 1
  rm -r "$lane/work/${w}_layers"
  bash "$E/run_fl343_models.sh" "$lane/work/$w.zarr" "$lane/ink/$w" 1 "${models[@]}" > /dev/null
  fwd=(); rev=()
  for model in "${models[@]}"; do
    for map in "$lane/ink/$w"/*__"${model}"__L66s26.tif; do [ -f "$map" ] && fwd+=("$map"); done
    for map in "$lane/ink/$w"/*__"${model}"__L66s26_reverse.tif; do [ -f "$map" ] && rev+=("$map"); done
  done
  if [ "${#fwd[@]}" -gt 0 ] && [ "${#rev[@]}" -gt 0 ]; then
    "$POSTEL/ink/.venv/bin/python" "$E/lane_layout.py" "$lane" "$w" "$lane/work/$w.npy" \
      --fwd "${fwd[@]}" --rev "${rev[@]}"
  else
    log "$w: no ink maps (see $lane/ink/$w/*.log)"
  fi
  rm -rf "$lane/work/$w.zarr" "$lane/work/$w.npy"
  log "$w done; laid out so far: $(find "$lane/maps" -name done_d0 2> /dev/null | wc -l)"
done
log "done"
