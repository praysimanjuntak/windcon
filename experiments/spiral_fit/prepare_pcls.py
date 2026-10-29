#!/usr/bin/env python3
"""Build the point-collection documents for the spiral-fit A/B experiment.

Writes into --out:
  heldout_same.json  human same-winding arcs drawn in single slices at the evaluation slices
                     (never a fit input; used only to score fitted checkpoints)
  human_train.json   every other human same-winding collection
  auto.json          all automatic paths (auto_z*.json in --auto-dir), merged
  human_plus_auto.json  human_train + auto
  empty.json         a document without collections
Collection and point ids are renumbered so merged documents have no collisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VERSION = "1"


def load(path: Path) -> list[dict]:
    doc = json.loads(path.read_text())
    assert doc.get("vc_pointcollections_json_version") == VERSION, path
    return [doc["collections"][k] for k in sorted(doc["collections"], key=int)]


def flat_z(col: dict) -> int | None:
    """Rounded z if the collection is drawn within one slice and has no winding annotations."""
    pts = list(col.get("points", {}).values())
    if len(pts) < 2 or any(p.get("wind_a") is not None for p in pts):
        return None
    zs = [p["p"][2] for p in pts]
    return round(zs[0]) if max(zs) - min(zs) <= 0.5 else None


def write(path: Path, cols: list[dict]) -> int:
    out, next_point = {}, 0
    for cid, col in enumerate(cols):
        items = sorted(col.get("points", {}).items(), key=lambda kv: int(kv[0]))
        points = {}
        for _, pt in items:
            points[str(next_point)] = pt
            next_point += 1
        out[str(cid)] = {k: v for k, v in col.items() if k != "points"} | {"points": points}
    path.write_text(json.dumps({"vc_pointcollections_json_version": VERSION, "collections": out}))
    return next_point


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--human", required=True, type=Path, help="human same_windings.json")
    ap.add_argument("--auto-dir", required=True, type=Path)
    ap.add_argument("--heldout-z", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    human = load(args.human)
    heldout = [c for c in human if flat_z(c) in set(args.heldout_z)]
    train = [c for c in human if flat_z(c) not in set(args.heldout_z)]
    auto = []
    for f in sorted(args.auto_dir.glob("auto_z*.json")):
        if f.name.endswith(".stats.json"):
            continue
        auto.extend(load(f))

    report = {}
    for name, cols in [("heldout_same", heldout), ("human_train", train), ("auto", auto),
                       ("human_plus_auto", train + auto), ("empty", [])]:
        report[name] = {"collections": len(cols), "points": write(args.out / f"{name}.json", cols)}
    report["heldout_by_z"] = {str(z): sum(flat_z(c) == z for c in heldout) for z in args.heldout_z}
    (args.out / "pcl_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main())
