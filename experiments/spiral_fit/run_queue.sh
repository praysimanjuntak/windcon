#!/usr/bin/env bash
# Train runs two at a time, score each on the held-out data, then re-collect every run's metrics.
#   [ROOT=<experiment root>] run_queue.sh <runA> <runB> [<runC> <runD> ...]
# A run is <regime>_s<seed> with datasets/<regime> and configs/<run>.json from setup_ab.py, all
# under ROOT (default $POSTEL; a second window lives in its own root).
# Finished steps are skipped, so the queue can be re-run after an interruption.
set -uo pipefail
source "$HOME/Documents/postel/vesuvius_challenge/env.sh"
ROOT="${ROOT:-$POSTEL}"; export ROOT
C="$ROOT/configs"; D="$ROOT/datasets"; R="$ROOT/runs"
S="$POSTEL/windcon/experiments/spiral_fit"
log() { echo "$(date '+%F %T') $*"; }
ok() { [ -f "$1/DONE" ] && grep -q '^exit=0' "$1/DONE"; }

train() {  # <run>
  local root="${1%_s*}"
  if ok "$R/$1"; then log "skip train $1"; return; fi
  rm -rf "$R/$1"
  log "train $1"
  "$POSTEL/tools/run_fit.sh" "$D/$root" "$R/$1" "$C/$1.json" > /dev/null 2>&1
  log "train $1: $(cat "$R/$1/DONE" 2>/dev/null || echo FAILED)"
}

evaluate() {  # <run> <patches|pcls>
  local out="$R/$1/eval_$2" root="$D/eval"
  case "$1" in
    few_rand*|few_targ*|few_orac*) root="$D/eval_al" ;;  # annotation targeting: common held-out set
    few_*) root="$D/eval_few" ;;  # the few family holds out 90 % of the patches
  esac
  if ! ok "$R/$1"; then log "skip eval $1 $2: training did not finish"; return; fi
  if ok "$out"; then log "skip eval $1 $2"; return; fi
  rm -rf "$out"
  FIT_SPIRAL_RESUME_PATH="$R/$1/checkpoint_fitted.ckpt" \
    "$POSTEL/tools/run_fit.sh" "$root" "$out" "$C/$1_eval_$2.json" > /dev/null 2>&1
  rm -f "$out/checkpoint_fitted.ckpt"  # zero-step resume: identical to the trained checkpoint
  log "eval $1 $2: $(cat "$out/DONE" 2>/dev/null || echo FAILED)"
}

log "queue start: $*"
runs=("$@")
for ((i = 0; i < ${#runs[@]}; i += 2)); do
  pair=("${runs[@]:i:2}")
  for run in "${pair[@]}"; do train "$run" & done
  wait
  for run in "${pair[@]}"; do
    case "$run" in sparse_*|few_*) evaluate "$run" patches ;; esac  # held out only in these families
    evaluate "$run" pcls
    if ok "$R/$run" && [ ! -f "$R/$run/diag_scanfree/satisfied_fitted.json" ]; then
      bash "$S/diag_eval.sh" "$run"
    fi
  done
done
shopt -s nullglob  # a root may hold only some families
cd "$POSTEL/windcon" && .venv/bin/python "$S/collect_metrics.py" "$R"/sparse_*_s? "$R"/full*_*_s? "$R"/few_*_s? \
  --heldout "$ROOT/pcls/heldout_same.json" --out "$ROOT/results_ab.json" > /dev/null
log "queue done"
