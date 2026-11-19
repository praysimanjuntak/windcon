#!/usr/bin/env bash
# Winding-model rays for a scroll that has none, and a spiral fit that uses them:
#   rays_for_scroll.sh <dataset> <tracks-only run> <out> <masked scan zarr URL> <fit level> <voxel um> [weight]
#
# <voxel um> is the voxel size at <fit level>: 9.596 for level 2 of a 2.399 um scan, 9.362 for level 0 of a
# 9.362 um scan.
#
# <dataset> is a fit_spiral.py dataset (spiral-scroll.json, umbilicus.json, tracks/, lasagna_inputs/) and
# <tracks-only run> a finished fit of it with tracks and normals (run_fit.sh's run folder). The steps:
#   1. rays: build_store.py seeds a ray slab every 96 voxels on every 8th winding of that fit (from winding 10) and
#      runs villa's winding model (scrollprize/winding_model_9um) on the scan, sampled every STEP voxels. The model
#      was trained on 9.6 um voxels, so STEP defaults to 9.6 / <voxel um> (other steps count worse: README);
#   2. outer shell: fit_spiral.py takes rays only with an outer shell, and no prize scroll has a published one.
#      make_outer_shell.py reads it off the scan's mask (skipped when the dataset has outer_shell/);
#   3. <out>/dataset: the dataset's inputs linked, plus winding_inference/ and outer_shell/;
#   4. <out>/rays.json: the run's overrides with rays on, the shell's own loss off, and shell_min_confidence 0.
#      Without that, fit_spiral.py drops every ray pair outside the part of the shell table it is sure of.
#      The ray weights (loss_weight_dense_spacing and _density) are [weight] (default 120: at the default 12 a
#      fit of an 8.64 um scan does not follow its rays);
#   5. the fit: run_fit.sh <out>/dataset <out>/run <out>/rays.json (set NO_FIT=1 to stop before it).
# Needs $POSTEL/env.sh, villa's spiral-fitting environment (run.sh), the winding model in $MODEL (default
# $POSTEL/wi/model/ckpt_final.pth) and $POSTEL/tools/run_fit.sh. Measured results: README, "The winding count
# and the winding sense on other scrolls".
set -euo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
dataset=$(cd "$1" && pwd); run=$(cd "$2" && pwd); out=$3; url=${4%/}; level=$5; voxel_um=$6; weight=${7:-120}
W="$POSTEL/windcon/experiments/winding_inference"; S="$POSTEL/windcon/experiments/spiral_fit"
MODEL=${MODEL:-$POSTEL/wi/model/ckpt_final.pth}
STEP=${STEP:-$(python3 -c "print(round(9.6 / $voxel_um, 4))")}
mkdir -p "$out"; out=$(cd "$out" && pwd)
read -r z0 z1 < <(python3 -c "import json; o = json.load(open('$run/overrides.json')); print(o['z_begin'], o['z_end'])")
log() { echo "$(date '+%F %T') $*"; }

cache=$("$POSTEL/windcon/.venv/bin/python" -c "
import sys; sys.path.insert(0, '$POSTEL/windcon/src')
from windcon.zarr_io import ZarrArray
print(ZarrArray('$url/$level').cache)")
if [ ! -f "$out/store/manifest.json" ]; then
  log "rays: step $STEP voxels, z $z0-$z1"
  bash "$W/run.sh" build_store.py --checkpoint "$run/checkpoint_fitted.ckpt" --umbilicus "$dataset/umbilicus.json" \
    --volume "$cache" --model "$MODEL" --out "$out/store" --winding-step 8 --seed-spacing 96 --columns 3 \
    --step "$STEP" --batch-size 1 --extract-threads 8
fi
shell="$dataset/outer_shell"
if [ ! -d "$shell" ]; then
  shell="$out/outer_shell"
  [ -d "$shell" ] || "$POSTEL/windcon/.venv/bin/python" "$S/make_outer_shell.py" --volume "$url" --mask-level 5 \
    --fit-level "$level" --umbilicus "$dataset/umbilicus.json" --z-begin "$z0" --z-end "$z1" --out "$shell"
fi
mkdir -p "$out/dataset"
for item in "$dataset"/*; do
  case "$(basename "$item")" in winding_inference|outer_shell|spiral-scroll.json) continue ;; esac
  ln -sfn "$item" "$out/dataset/$(basename "$item")"
done
cp "$dataset/spiral-scroll.json" "$out/dataset/spiral-scroll.json"
ln -sfn "$out/store" "$out/dataset/winding_inference"
ln -sfn "$shell" "$out/dataset/outer_shell"
python3 - "$run/overrides.json" "$out/rays.json" "$weight" <<'PY'
import json, sys
overrides = json.load(open(sys.argv[1]))
weight = float(sys.argv[3])
overrides.update({
    "input_use_winding_inference": True,
    "input_use_outer_shell": True,
    "input_use_gradient_magnitude": False,
    "dense_spacing_mode": "winding_model",
    "loss_weight_shell_outer": 0.0,
    "shell_min_confidence": 0.0,
    "loss_weight_dense_spacing": weight,
    "loss_weight_dense_spacing_density": weight,
})
json.dump(overrides, open(sys.argv[2], "w"), indent=1)
PY
log "dataset $out/dataset, overrides $out/rays.json (weight $weight)"
[ "${NO_FIT:-0}" = 1 ] && exit 0
"$POSTEL/tools/run_fit.sh" "$out/dataset" "$out/run" "$out/rays.json"
