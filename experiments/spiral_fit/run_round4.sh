#!/usr/bin/env bash
# Round 4, after every queued training run: score the automatic paths as single-slice paths and
# as chains against the given runs (agree_eval.sh -> agree.json), and split every run's held-out
# scores by verified-patch coverage (coverage_split.py -> coverage.json).
# Usage: bash run_round4.sh <run> <run> ...
set -uo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
S="$POSTEL/windcon/experiments/spiral_fit"; R="$POSTEL/runs"
log() { echo "$(date '+%F %T') $*"; }

# The scoring passes need the card to themselves next to two training fits: wait for the queues.
while pgrep -f "run_round[23].sh|run_queue.sh" > /dev/null; do sleep 30; done
log "agree"
bash "$S/agree_eval.sh" "$@"
cd "$POSTEL/windcon"
uv run python "$S/agree_summary.py" "${@/#/$R/}" --chains-dir "$POSTEL/pcls" --out "$POSTEL/agree.json"
log "coverage"
if [ ! -f "$POSTEL/fiber_points.json" ]; then
  (cd "$POSTEL/villa/spiral-fitting" && .venv/bin/python "$S/dump_fibers.py" \
    --fibers "$POSTEL/datasets/eval/fibers" --out "$POSTEL/fiber_points.json")
fi
uv run python "$S/coverage_split.py" "$R"/sparse_?_s? "$R"/full*_?_s? --dataset "$POSTEL/data/PHercParis4" \
  --heldout "$POSTEL/pcls/heldout_same.json" --fibers "$POSTEL/fiber_points.json" --out "$POSTEL/coverage.json"
log "round 4 done"
