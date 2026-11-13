#!/usr/bin/env bash
# The four ink models of the PHerc. 343 First Letters submission on a 66-layer render, at one
# depth, through the submission's own run_ink.sh (github.com/Nieuwlaar/pherc343-first-letters):
#   run_fl343_models.sh <render.zarr> <out dir> [depth, default 1] [model ...]
# Models: dnative hybrid_3d2d-seed42 hybrid_3d2d-seed43 hecate (default: all four).
# Needs the submission's repository in $POSTEL/data/PHerc0343/winner, its checkpoints in
# $POSTEL/ink/ckpt343 (ink/scripts/fetch_checkpoints.sh ink/models.tsv $POSTEL/ink/ckpt343),
# $POSTEL/ink/.venv and villa's vesuvius package. The submission ran villa 0fb38c45; this runs 5a4388f.
set -uo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
render="$1"; out="$2"; depth="${3:-1}"; shift; shift; shift || true
models=("$@"); [ "${#models[@]}" -eq 0 ] && models=(dnative hybrid_3d2d-seed42 hybrid_3d2d-seed43 hecate)
export PYTHONPATH="$POSTEL/villa/vesuvius/src"
# The models' data loaders open sockets under TMPDIR, whose path may not exceed 108 characters.
export CKPT_DIR="$POSTEL/ink/ckpt343" PY="$POSTEL/ink/.venv/bin/python" TMPDIR="$POSTEL/t"
mkdir -p "$TMPDIR"
export VILLA_BATCH="${VILLA_BATCH:-4}" HEC_BATCH="${HEC_BATCH:-4}"
mkdir -p "$out"
for model in "${models[@]}"; do
  bash "$POSTEL/data/PHerc0343/winner/ink/scripts/run_ink.sh" "$render" "$model" "$depth" "$out" \
    > "$out/${model}_d${depth}.log" 2>&1 || echo "FAILED $model"
  echo "$(date '+%T') $model d=$depth done"
done
