#!/usr/bin/env bash
# Score finished runs on datasets/diag (automatic paths + held-out arcs + eval fibers + relative
# windings), once with the fitter's default satisfaction tolerances and once without its 6-voxel
# scan-distance test:  [ROOT=<experiment root>] diag_eval.sh <run> [<run> ...]
set -uo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
ROOT="${ROOT:-$POSTEL}"
R="$ROOT/runs"
for run in "$@"; do
  for mode in default scanfree; do
    out="$R/$run/diag_$mode"
    rm -rf "$out"; mkdir -p "$out"
    (
      cd "$POSTEL/villa/spiral-fitting"
      export FIT_SPIRAL_CONFIG_OVERRIDES="$(cat "$ROOT/configs/${run}_eval_diag.json")"
      export FIT_SPIRAL_RUN_DIR="$out" FIT_SPIRAL_SKIP_SAVE_MESH=1 PYTHONUNBUFFERED=1
      export FIT_SPIRAL_RESUME_PATH="$R/$run/checkpoint_fitted.ckpt"
      if [ "$mode" = scanfree ]; then export SAT_SCAN_TOL=1e9; fi
      .venv/bin/python "$POSTEL/windcon/experiments/spiral_fit/eval_relaxed.py" \
        --dataset "$ROOT/datasets/diag" > "$out/fit.log" 2>&1
    )
    rm -f "$out/checkpoint_fitted.ckpt"
    echo "$(date '+%F %T') diag $run $mode: $(grep -E 'satisfied_unattached_pcl_points' "$out/fit.log")"
  done
done
