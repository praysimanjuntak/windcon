#!/usr/bin/env bash
# Score finished runs on automatic paths alone (datasets/agree_<key>, no patches, so every path,
# chain or relative-winding pair is one unattached strip), winding test only (no 6-voxel scan test):
#   [ROOT=<experiment root>] [AGREE_KEYS="singles chains relwind"] agree_eval.sh <run> [<run> ...]
# Output: runs/<run>/agree_<key>/satisfied_fitted.json; summarise with agree_summary.py.
set -uo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
ROOT="${ROOT:-$POSTEL}"
R="$ROOT/runs"
for run in "$@"; do
  for key in ${AGREE_KEYS:-singles chains}; do
    out="$R/$run/agree_$key"
    if [ -f "$out/satisfied_fitted.json" ]; then continue; fi
    rm -rf "$out"; mkdir -p "$out"
    (
      cd "$POSTEL/villa/spiral-fitting"
      export FIT_SPIRAL_CONFIG_OVERRIDES="$(cat "$ROOT/configs/${run}_eval_agree.json")"
      export FIT_SPIRAL_RUN_DIR="$out" FIT_SPIRAL_SKIP_SAVE_MESH=1 PYTHONUNBUFFERED=1
      export FIT_SPIRAL_RESUME_PATH="$R/$run/checkpoint_fitted.ckpt" SAT_SCAN_TOL=1e9
      .venv/bin/python "$POSTEL/windcon/experiments/spiral_fit/eval_relaxed.py" \
        --dataset "$ROOT/datasets/agree_$key" > "$out/fit.log" 2>&1
    )
    rm -f "$out/checkpoint_fitted.ckpt"
    echo "$(date '+%F %T') agree $run $key: $(grep -E 'satisfied_unattached_pcl_points' "$out/fit.log")"
  done
done
