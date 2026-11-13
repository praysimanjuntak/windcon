#!/usr/bin/env bash
# Which way does a scroll wind? From its surface tracks alone, before any fit.
#   winding_sense.sh <tracks .dbm> <umbilicus.json> <z_begin> <z_end> <inner radius> <outer radius> <out dir> [seeds] [workers]
# The .dbm must have its packed .vctracks folder and .crossings.npz beside it (villa's
# convert_track_store.py and build_track_crossings.py). Seeds are long tracks whose median
# distance from the umbilicus lies between the two radii (pick_track_seeds.py); villa's
# grow_track_graph.py grows a patch from each with the settings of the published band-seed
# patches and a size limit of 40 cm2, and patch_sense.py reads the sense off the patches that pass
# a full turn. Near the umbilicus a turn is short, so seed there: 64 seeds between a radius of 350
# and 700 voxels gave 10 such patches on PHerc. 0139, in three minutes on eight cores.
# MIN_SIZE (cm2, default 0.5) is the smallest patch kept; lower it with the radius.
set -uo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
tracks="$1"; umbilicus="$2"; z0="$3"; z1="$4"; r0="$5"; r1="$6"; out="$7"; count="${8:-64}"; workers="${9:-8}"
S="$POSTEL/windcon/experiments/spiral_fit"
mkdir -p "$out/patches"
cd "$POSTEL/villa/spiral-fitting"
seeds=$(.venv/bin/python "$S/pick_track_seeds.py" "$tracks.vctracks" --z "$z0" "$z1" --count "$count" \
  --umbilicus "$umbilicus" --radius "$r0" "$r1")
# shellcheck disable=SC2086  # one argument per seed row
.venv/bin/python grow_track_graph.py "$tracks.vctracks" "$tracks.crossings.npz" "$out/patches" --seeds $seeds \
  --min-connect 6 --growth-min-span 60 --min-size "${MIN_SIZE:-0.5}" --max-size 40 --workers "$workers" > "$out/grow.log" 2>&1
tail -n 1 "$out/grow.log"
"$POSTEL/windcon/.venv/bin/python" "$S/patch_sense.py" --umbilicus "$umbilicus" --patches "$out/patches" --out "$out/sense.json"
