#!/usr/bin/env python3
"""Summarise spiral-fit A/B runs: training-input satisfaction and held-out evaluation passes.

Each trained run directory may contain:
  eval_patches/  zero-step resume with the verified patches enabled (held out in the sparse regime)
  eval_pcls/     zero-step resume with only point collections enabled (held-out arcs, eval fibers,
                 relative windings; the latter are training inputs in the full regime)
Prints one JSON object per run and writes them to --out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def last_step_line(fit_log: Path) -> str:
    lines = [ln for ln in fit_log.read_text(errors="replace").replace("\r", "\n").splitlines()
             if ln.startswith("step ")]
    return lines[-1] if lines else ""


GROUP_BY_FILE = {
    "relative_windings.json": "relative",
    "same_windings.json": "same_winding",  # held-out arcs in eval passes, human_train in training
    "drawn_control_points.json": "auto",
}


def pcl_groups(entries: list[dict], heldout_z: dict[str, int]) -> dict:
    groups: dict[str, list[dict]] = {}
    for e in entries:
        src = str(e.get("source_file") or "")
        key = "fibers" if "/fibers/" in src else GROUP_BY_FILE.get(Path(src).name, "other")
        groups.setdefault(key, []).append(e)
        if key == "same_winding" and e.get("name") in heldout_z:
            groups.setdefault(f"same_winding_z{heldout_z[e['name']]}", []).append(e)
    out = {}
    for key, es in sorted(groups.items()):
        pts = sum(e["total_points"] for e in es)
        out[key] = {
            "collections": len(es),
            "fully_satisfied": sum(e["satisfied_points"] == e["total_points"] for e in es),
            "fully_satisfied_fraction": round(
                sum(e["satisfied_points"] == e["total_points"] for e in es) / max(len(es), 1), 4),
            "points": pts,
            "points_satisfied_fraction": round(sum(e["satisfied_points"] for e in es) / max(pts, 1), 4),
        }
    return out


def summarise(run: Path, heldout_z: dict[str, int]) -> dict:
    res: dict = {"run": run.name}
    done = run / "DONE"
    if done.exists():
        res["train"] = done.read_text().strip()
    if (run / "fit.log").exists():
        res["final_step"] = last_step_line(run / "fit.log")
    tm = run / "satisfaction_metrics_fitted.json"
    if tm.exists():  # satisfaction of the run's own training inputs
        s = json.loads(tm.read_text())["summary"]
        res["train_inputs"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()
                               if k.endswith("_fraction") or k.startswith("total_")}
    ts = run / "satisfied_fitted.json"
    if ts.exists():
        res["train_pcls"] = pcl_groups(json.loads(ts.read_text())["pcls"], {})
    ev = run / "eval_patches" / "satisfaction_metrics_fitted.json"
    if ev.exists():
        s = json.loads(ev.read_text())["summary"]
        res["patches"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()
                          if "patch" in k or k.startswith(("satisfied_area", "total_area"))}
    ev = run / "eval_pcls" / "satisfied_fitted.json"
    if ev.exists():
        res["pcls"] = pcl_groups(json.loads(ev.read_text())["pcls"], heldout_z)
    for mode in ("default", "scanfree"):  # diag_eval.sh passes
        ev = run / f"diag_{mode}" / "satisfied_fitted.json"
        if ev.exists():
            res[f"diag_{mode}"] = pcl_groups(json.loads(ev.read_text())["pcls"], heldout_z)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--heldout", required=True, type=Path, help="heldout_same.json")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    doc = json.loads(args.heldout.read_text())["collections"]
    heldout_z = {c["name"]: round(next(iter(c["points"].values()))["p"][2]) for c in doc.values()}
    rows = [summarise(r, heldout_z) for r in args.runs if r.is_dir()]
    args.out.write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    sys.exit(main())
