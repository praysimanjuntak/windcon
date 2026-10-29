#!/usr/bin/env bash
# Gap-filled path set (pcls_g/): every 5th slice of the window, including the 20 slices within 25
# of the held-out arcs that rounds 2-3 left out, so the arcs sit 1-4 slices from traced paths like
# every other location. Builds chains and the no-patch variant, the own-budget overrides, checks
# the links, then trains and scores the path variants (the no-path baselines are unchanged).
# Resumable: finished steps are skipped.   bash run_gapfill.sh [<run> ...]
set -uo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
S="$POSTEL/windcon/experiments/spiral_fit"
DATA="$POSTEL/data/PHercParis4"; OUT="$POSTEL/pcls_g"
log() { echo "$(date '+%F %T') $*"; }
cd "$POSTEL/windcon"

while pgrep -f "trace_window[.]py|find_adjacency[.]py" > /dev/null; do sleep 30; done
n_raw=$(ls "$POSTEL"/auto_paths_z5/auto_z*.npz 2>/dev/null | wc -l)
if [ "$n_raw" -ne 199 ]; then log "expected 199 traced slices, found $n_raw"; exit 1; fi

if [ ! -f "$OUT/auto_chains.json" ]; then
  log "build chains"
  uv run python "$S/build_chains.py" --raw-dir "$POSTEL/auto_paths_z5" --umbilicus "$DATA/umbilicus.json" \
    --out "$OUT" --delta 1.9 || exit 1
fi
if [ ! -f "$OUT/auto_chains_free.json" ]; then
  log "drop near-patch points"
  uv run python "$S/drop_near_patches.py" --pcl "$OUT/auto_chains.json" --dataset "$DATA" \
    --out "$OUT/auto_chains_free.json" || exit 1
fi

python3 "$S/setup_ab.py" --postel "$POSTEL" > /dev/null
for regime in fullnodt_Eg fullnodt_Gg; do
  if [ ! -f "$POSTEL/configs/own_budget_$regime.json" ]; then
    log "own budget $regime"
    (cd "$POSTEL/villa/spiral-fitting" && FIT_SPIRAL_CONFIG_OVERRIDES="$(cat "$POSTEL/configs/${regime}_s1_base.json")" \
      .venv/bin/python "$S/sample_shares.py" --dataset "$POSTEL/datasets/$regime" \
      --emit "$POSTEL/configs/own_budget_$regime.json" > "$POSTEL/logs/shares_$regime.log" 2>&1) || exit 1
    log "$regime: $(tr -d '\n ' < "$POSTEL/configs/own_budget_$regime.json")"
  fi
done
python3 "$S/setup_ab.py" --postel "$POSTEL" > /dev/null

if [ ! -f "$OUT/check_links.json" ]; then
  log "check links"
  uv run python "$S/check_links.py" --chains-dir "$OUT" --dataset "$DATA" | tail -n 18
fi

# Never three fits on the 12 GB card: wait for any other queue to finish.
while pgrep -f "run_queue[.]sh" > /dev/null; do sleep 30; done
runs=("$@")
[ ${#runs[@]} -eq 0 ] && runs=(fullnodt_Eg_s1 sparse_Eg_s1 fullnodt_Eg_s2 sparse_Eg_s2 fullnodt_Ig_s1 fullnodt_Gg_s1)
log "train ${runs[*]}"
bash "$S/run_queue.sh" "${runs[@]}"
log "coverage"
uv run python "$S/coverage_split.py" "$POSTEL"/runs/sparse_*_s? "$POSTEL"/runs/full*_*_s? --dataset "$DATA" \
  --heldout "$POSTEL/pcls/heldout_same.json" --fibers "$POSTEL/fiber_points.json" --out "$POSTEL/coverage.json" \
  | tail -n 40
log "gap-fill done"
