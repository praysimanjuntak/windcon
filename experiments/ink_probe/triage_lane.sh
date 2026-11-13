#!/usr/bin/env bash
# The PHerc. 0306B First Letters submission's triage, unchanged, on a lane written by lane_layout.py:
#   triage_lane.sh <lane folder> <voxel size in um> [label for the cards]
# Runs its window_score.py (row-periodicity score of every window 2,600 px wide, every 1,300 px,
# both faces), tri_rank.py (windows with a row score of $MIN or more, default its own 4, through
# the hard and soft gates; the submission's lane ran it with -99, every window) and
# cards_from_tri.py (a review card for each REVIEW window and the first $REVIEW2 (10) of the second
# tier). Needs the submission's repository with discovery/{fl,x_vast_plan,CANDIDATES} checked out
# in $POSTEL/data/PHerc0306B_winner and $POSTEL/ink/.venv (numpy, scipy, opencv, zarr).
# With GLYPH=1 its glyph network scores every window too (glyph_win.py, on the CPU, about 40 s a
# winding) and tri_rank.py also admits windows at or over that score's own threshold. Two things
# are changed for that, in a copy under $POSTEL/ink/glyph_root: the work folder hard-coded in
# x_glyph/common.py, and scikit-image, which only the synthetic training letters use, is stubbed.
# Outputs: <lane>/maps/*/spectral_win.json (and glyph_win.json), <lane>/tri_rank.{json,txt},
# <lane>/cards/.
set -uo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
lane="$(cd "$1" && pwd)"; um="$2"; label="${3:-$(basename "$lane")}"
R="$POSTEL/data/PHerc0306B_winner/discovery"; py="$POSTEL/ink/.venv/bin/python"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" PYTHONPATH="$R/CANDIDATES/_tools"
(cd "$R/fl/remote/spiral" && "$py" window_score.py "$lane/maps" --um "$um") || exit 1
glyph=()
if [ "${GLYPH:-0}" = 1 ]; then
  G="$POSTEL/ink/glyph_root"
  if [ ! -f "$G/x_glyph/common.py" ]; then
    mkdir -p "$G/fl" "$G/shim/skimage"
    cp -r "$R/x_glyph" "$G/" && cp "$R/fl/scrolls.json" "$G/fl/"
    sed -i "s#^W = .*#W = '$G'#" "$G/x_glyph/common.py"
    : > "$G/shim/skimage/__init__.py"
    printf 'def skeletonize(*args, **kwargs):\n    raise NotImplementedError("scikit-image is not installed")\n' \
      > "$G/shim/skimage/morphology.py"
  fi
  (cd "$R/fl/scripts" && PYTHONPATH="$G/shim" FL_W="$G" CUDA_VISIBLE_DEVICES="" "$py" glyph_win.py "$lane/maps" \
    --um "$um" --model "$G/x_glyph/models/gA.pt") > "$lane/glyph_win.log" 2>&1 || { tail -n 5 "$lane/glyph_win.log"; exit 1; }
  glyph=(--glyph)
fi
(cd "$R/fl/scripts" && "$py" tri_rank.py "$lane/maps" --um "$um" --min "${MIN:-4}" "${glyph[@]}" --json "$lane/tri_rank.json") \
  > "$lane/tri_rank.txt" 2> "$lane/tri_rank.err" || { tail -n 5 "$lane/tri_rank.err"; exit 1; }
echo "$(grep -c -- '-> REVIEW$' "$lane/tri_rank.txt") windows in REVIEW, $(grep -c -- '-> REVIEW-2' "$lane/tri_rank.txt") in REVIEW-2, $(grep -c -- '-> auto-reject' "$lane/tri_rank.txt") rejected"
rm -rf "$lane/cards"
(cd "$R/x_vast_plan" && "$py" cards_from_tri.py "$lane/tri_rank.json" "$lane/maps" "$lane/renders" "$lane/cards" "$um" \
  --review2 "${REVIEW2:-10}" --label "$label") | tail -n 1
