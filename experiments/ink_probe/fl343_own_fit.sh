#!/usr/bin/env bash
# PHerc. 343 control, second part: do windings of a fit made here show the letters of the First
# Letters submission (github.com/Nieuwlaar/pherc343-first-letters, in $POSTEL/data/PHerc0343/winner)?
#   fl343_own_fit.sh <folder of winding meshes> <name> <winding> [<winding> ...]
# <folder of winding meshes> holds w###/ tifxyz folders: <run>/meshes/fitted as fit_spiral.py writes
# them (FIT_SPIRAL_SKIP_SAVE_MESH unset), or the output of the submission's snap_meshes.py.
# For each winding: render 66 layers of the part within 200 voxels of the submission's mesh, with
# the normals turned away from the scroll's axis, and run the submission's four ink models at depth
# +1 (run_fl343_models.sh; $MODELS names fewer, Hecate takes 16 minutes a winding). The maps are
# then brought into the submission's pixel frame (resample_to_mesh.py) and scored as the maps of
# its own mesh are (fl343_check.py).
# Outputs: $POSTEL/ink/segments/PHerc0343/<name>_w###.{npy,zarr}, $POSTEL/ink/out/fl343_<name>/.
set -uo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
meshes="$1"; name="$2"; shift 2
E="$POSTEL/windcon/experiments/ink_probe"
D="$POSTEL/data/PHerc0343"; W="$D/winner"; S="$POSTEL/ink/segments/PHerc0343"; O="$POSTEL/ink/out/fl343_$name"
N=concat_w047-w048_R5B2_z9500-11000
CT=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr/0
py="$POSTEL/windcon/.venv/bin/python"
read -r -a models <<< "${MODELS:-dnative hybrid_3d2d-seed42 hybrid_3d2d-seed43 hecate}"
mkdir -p "$S" "$O/resampled"
cd "$POSTEL/windcon"
for w in "$@"; do
  stack="${name}_$w"
  if [ ! -d "$S/$stack.zarr" ]; then
    "$py" "$E/render_mesh.py" "$meshes/$w" --volume "$CT" --umbilicus "$D/umbilicus.json" --layers 66 \
      --near "$W/outputs/$N.tifxyz" --within 200 --out "$S/${stack}_layers" | tail -n 2
    "$py" "$E/fetch_layers.py" "$S/${stack}_layers" --name "$stack" --native-spacing 8.64 --out "$S" --zarr
    rm -r "$S/${stack}_layers"
  fi
  bash "$E/run_fl343_models.sh" "$S/$stack.zarr" "$O/$w" 1 "${models[@]}"
done
# One map per model in the submission's frame, each pixel from the winding nearest to its mesh.
for model in "${models[@]}"; do
  sources=()
  for w in "$@"; do
    map=$(ls "$O/$w"/*__"${model}"__*_reverse.* 2>/dev/null | grep -v '\.json$' | head -n 1)
    [ -n "$map" ] && sources+=(--source "$meshes/$w" "$map")
  done
  [ "${#sources[@]}" -gt 0 ] || { echo "no maps of $model"; continue; }
  "$py" "$E/resample_to_mesh.py" --target "$W/outputs/$N.tifxyz" "${sources[@]}" \
    --out "$O/resampled/${name}__${model}__resampled_reverse" | tail -n 3
done
"$py" "$E/fl343_check.py" --maps "$O/resampled" --render "$S/winner66.npy" --sidecar "$W/outputs/$N.json" \
  --image "$W/outputs/$N.png" --out "$O/$name" | tee "$O/scores.json"
