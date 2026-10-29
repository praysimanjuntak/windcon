#!/usr/bin/env bash
# Round 2 of the spiral-fit experiment, after trace_window.py has written auto_paths_z5/:
# build smoothed, shifted, linked paths; drop near-patch points for the no-attachment variant;
# compute the own-budget overrides; set up the regimes; verify the paths; train and score.
# Resumable: finished steps are skipped (files exist, runs have DONE).
#   MIN_CHAIN_PATHS=1 DELTA=1.9 bash run_round2.sh
set -uo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
S="$POSTEL/windcon/experiments/spiral_fit"
DATA="$POSTEL/data/PHercParis4"; PCLS="$POSTEL/pcls"
log() { echo "$(date '+%F %T') $*"; }
cd "$POSTEL/windcon"


while pgrep -f "trace_window.py" > /dev/null; do sleep 30; done
n_raw=$(ls "$POSTEL"/auto_paths_z5/auto_z*.npz 2>/dev/null | wc -l)
if [ "$n_raw" -ne 179 ]; then log "expected 179 traced slices, found $n_raw: re-run trace_window.py"; exit 1; fi
if [ ! -f "$PCLS/auto_chains.json" ]; then
  log "build chains"
  uv run python "$S/build_chains.py" --raw-dir "$POSTEL/auto_paths_z5" --umbilicus "$DATA/umbilicus.json" \
    --out "$PCLS" --delta "${DELTA:-1.9}" --min-chain-paths "${MIN_CHAIN_PATHS:-1}" || exit 1
fi
if [ ! -f "$PCLS/auto_chains_free.json" ]; then
  log "drop near-patch points"
  uv run python "$S/drop_near_patches.py" --pcl "$PCLS/auto_chains.json" --dataset "$DATA" \
    --out "$PCLS/auto_chains_free.json" || exit 1
fi

python3 "$S/setup_ab.py" --postel "$POSTEL"
for regime in fullnodt_E fullnodt_G; do
  if [ ! -f "$POSTEL/configs/own_budget_$regime.json" ]; then
    log "own budget $regime"
    (cd "$POSTEL/villa/spiral-fitting" && FIT_SPIRAL_CONFIG_OVERRIDES="$(cat "$POSTEL/configs/${regime}_s1_base.json")" \
      .venv/bin/python "$S/sample_shares.py" --dataset "$POSTEL/datasets/$regime" \
      --emit "$POSTEL/configs/own_budget_$regime.json" > "$POSTEL/logs/shares_$regime.log" 2>&1) || exit 1
    log "$regime: $(cat "$POSTEL/configs/own_budget_$regime.json" | tr -d '\n ')"
  fi
done
python3 "$S/setup_ab.py" --postel "$POSTEL"

if [ ! -f "$PCLS/check_links.json" ]; then
  log "verify shift and links"
  uv run python "$S/offset_check.py" --dataset "$DATA" --pcl "$PCLS/auto_singles.json" "$PCLS/auto.json" \
    --z 10025 10425 10825 | tee "$POSTEL/logs/offset_round2.log"
  uv run python "$S/check_links.py" --chains-dir "$PCLS" --dataset "$DATA" | tail -n 25
fi

# Never three fits on the 12 GB card: wait for any other queue to finish.
while pgrep -f "run_queue.sh" > /dev/null; do sleep 30; done
log "train"
bash "$S/run_queue.sh" fullnodt_E_s1 fullnodt_G_s1 sparse_E_s1 sparse_E_s2 \
  fullnodt_E_s2 fullnodt_G_s2 sparse_F_s1 fullnodt_D_s2
log "round 2 done"
