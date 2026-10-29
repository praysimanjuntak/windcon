#!/usr/bin/env bash
# Second window (z 13400-14400) for the sparse-family test, in its own experiment root
# ($POSTEL/w2: data/, pcls/, pcls_g/, datasets/, configs/, runs/). Global inputs (umbilicus, outer
# shell, winding inference, eval fibres, human point collections, the patch meta.json cache) are
# symlinked from the first window; the normals rows and verified patches of the window are
# fetched. Paths are traced every 5th slice on a grid offset by 2 (13402, 13407, ...), so the
# held-out human arcs (z 13500, 14068) lie 1-3 slices from traced slices, as in round 4.
# Resumable.   bash run_window2.sh [<run> ...]
set -uo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
S="$POSTEL/windcon/experiments/spiral_fit"
W1="$POSTEL/data/PHercParis4"; W2="$POSTEL/w2"; D2="$W2/data/PHercParis4"
Z0=13400; Z1=14400; BBOX="800 7660 1260 7800"
log() { echo "$(date '+%F %T') $*"; }
cd "$POSTEL/windcon"

mkdir -p "$D2" "$W2/pcls"
for f in umbilicus.json abs_winding.json relative_windings.json same_windings.json \
         outer_shell winding_inference eval_fibers _patch_meta; do
  [ -e "$D2/$f" ] || ln -s "$W1/$f" "$D2/$f"
done
log "fetch normals and patches"
uv run python "$S/fetch_dataset.py" --out "$D2" --z-begin $Z0 --z-end $Z1 --parts normals patches \
  > "$POSTEL/logs/fetch_w2.log" 2>&1 || { log "fetch failed"; exit 1; }
log "$(tail -n 3 "$POSTEL/logs/fetch_w2.log" | tr '\n' ' ')"

zs=$(python3 -c "print(*range($Z0 + 2, $Z1, 5))")
log "trace $(echo $zs | wc -w) slices"
uv run python "$S/trace_window.py" --out "$W2/auto_paths_z5" --bbox $BBOX --workers 16 --z $zs \
  > "$POSTEL/logs/trace_w2.log" 2>&1 || { log "trace failed"; exit 1; }
if [ ! -f "$W2/pcls_g/auto_chains.json" ]; then
  log "build chains"
  uv run python "$S/build_chains.py" --raw-dir "$W2/auto_paths_z5" --umbilicus "$D2/umbilicus.json" \
    --out "$W2/pcls_g" --delta 1.9 | tail -n 1 || exit 1
fi
if [ ! -f "$W2/pcls/heldout_same.json" ]; then
  log "held-out arcs and raw paths"
  uv run python "$S/prepare_pcls.py" --human "$D2/same_windings.json" --auto-dir "$W2/auto_paths_z5" \
    --heldout-z 13500 14068 --out "$W2/pcls" | tail -n 8 || exit 1
fi
python3 "$S/setup_ab.py" --postel "$W2" --z-begin $Z0 --z-end $Z1 | tail -n 1

# Never three fits on the 12 GB card: wait for every other queue and pipeline.
while pgrep -f "run_queue[.]sh|run_gapfill[.]sh" > /dev/null; do sleep 30; done
runs=("$@")
[ ${#runs[@]} -eq 0 ] && runs=(sparse_A_s1 sparse_Eg_s1 sparse_A_s2 sparse_Eg_s2)
log "train ${runs[*]}"
ROOT="$W2" bash "$S/run_queue.sh" "${runs[@]}"
log "window 2 done"
