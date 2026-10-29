#!/usr/bin/env bash
# Run one headless spiral fit:  run_fit.sh <dataset_root> <run_dir> <overrides.json>
# With FIT_SPIRAL_RESUME_PATH set to a finished checkpoint and the same step count, the fit
# resumes at its last step, runs zero steps and only writes the satisfaction metrics of the
# root's inputs: that is how checkpoints are scored on held-out data.
set -uo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
root="$1"; run_dir="$2"; overrides="$3"
mkdir -p "$run_dir"
cp "$overrides" "$run_dir/overrides.json"
cd "$POSTEL/villa/spiral-fitting"
export FIT_SPIRAL_CONFIG_OVERRIDES="$(cat "$overrides")"
export FIT_SPIRAL_RUN_DIR="$run_dir"
export FIT_SPIRAL_SKIP_SAVE_MESH="${FIT_SPIRAL_SKIP_SAVE_MESH:-1}"
export PYTHONUNBUFFERED=1
# Whole-GPU memory in use, sampled every 0.5 s (includes any other process on the card).
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > "$run_dir/gpu_mem.log" &
smi=$!
trap 'kill $smi 2>/dev/null || true' EXIT
start=$(date +%s)
status=0
/usr/bin/time -v .venv/bin/python fit_spiral.py --dataset "$root" > "$run_dir/fit.log" 2>&1 || status=$?
echo "exit=$status elapsed=$(( $(date +%s) - start ))s peak_gpu_MiB=$(sort -n "$run_dir/gpu_mem.log" | tail -n 1)" \
  | tee "$run_dir/DONE"
exit "$status"
