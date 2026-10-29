#!/usr/bin/env bash
# Round 3: ablations and extra seeds for round 2, queued behind any running queue.
#   sparse_H  unshifted singles (isolates the shift; compare sparse_F)
#   fullnodt_I  chains with the fitter's default budget (isolates the own budget; compare fullnodt_E)
# Usage: bash run_round3.sh <run> <run> ...   (e.g. sparse_H_s1 fullnodt_I_s1 fullnodt_A_s3 fullnodt_E_s3)
set -uo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
S="$POSTEL/windcon/experiments/spiral_fit"
log() { echo "$(date '+%F %T') $*"; }
cd "$POSTEL/windcon"

if [ ! -f "$POSTEL/pcls/auto_singles_noshift.json" ]; then
  log "build unshifted singles"
  out="$POSTEL/scratch/noshift"; mkdir -p "$out"
  uv run python "$S/build_chains.py" --raw-dir "$POSTEL/auto_paths_z5" \
    --umbilicus "$POSTEL/data/PHercParis4/umbilicus.json" --out "$out" --delta 0 || exit 1
  mv "$out/auto_singles.json" "$POSTEL/pcls/auto_singles_noshift.json"
fi
python3 "$S/setup_ab.py" --postel "$POSTEL"

# Never three fits on the 12 GB card: wait for any other queue to finish.
while pgrep -f "run_queue.sh" > /dev/null; do sleep 30; done
log "train"
bash "$S/run_queue.sh" "$@"
log "round 3 done"
