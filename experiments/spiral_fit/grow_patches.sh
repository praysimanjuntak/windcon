#!/usr/bin/env bash
# Grow patches from the surface tracks inside a z-window with villa's grow_track_graph.py and the
# settings recorded in the published band-seed patches' meta.json (min_connect 6, growth_min_span 60,
# sizes 0.05-10 cm2). Nobody checks the result.
#   grow_patches.sh <z_begin> <z_end> <seed count> <out dir> [workers]
# The seeds are the first <seed count> in-window rows of the grower's own random order
# (pick_track_seeds.py). They are grown 25,000 at a time; a seed that grows too little writes
# nothing (about one in nine of window 1's seeds wrote a patch).
set -uo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
z0="$1"; z1="$2"; count="$3"; out="$4"; workers="${5:-10}"
T="$POSTEL/data/PHercParis4/tracks/2um_ds2_ps256_surf_v2.dbm"
S="$POSTEL/windcon/experiments/spiral_fit"
seeds="$POSTEL/scratch/grow_seeds_${z0}_${z1}"
cd "$POSTEL/villa/spiral-fitting"
.venv/bin/python "$S/pick_track_seeds.py" "$T.vctracks" --z "$z0" "$z1" --count "$count" \
  | tr ' ' '\n' | split -l 25000 - "$seeds.part."
for part in "$seeds".part.*; do
  # shellcheck disable=SC2046  # one argument per seed row
  .venv/bin/python grow_track_graph.py "$T.vctracks" "$T.crossings.npz" "$out" --seeds $(cat "$part") \
    --min-connect 6 --growth-min-span 60 --min-size 0.05 --max-size 10 --workers "$workers" \
    > "$POSTEL/logs/grow_$(basename "$part").log" 2>&1
  echo "$(date '+%T') $(basename "$part"): $(ls "$out" | wc -l) patches so far"
done
