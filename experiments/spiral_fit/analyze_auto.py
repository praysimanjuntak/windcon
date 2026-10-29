#!/usr/bin/env python3
"""Break down how a fitted spiral agrees with the automatic paths (diag_eval.sh output).

Groups the per-path satisfaction of the fitter's diag pass by path length, by distance from the
umbilicus and by the tracer's stop reasons.

    python analyze_auto.py --satisfied runs/full_A_s1/diag_scanfree/satisfied_fitted.json \
        --auto-dir auto_paths --umbilicus data/PHercParis4/umbilicus.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def umbilicus_xy(path: Path):
    pts = sorted((p["z"], p["x"], p["y"]) for p in json.loads(path.read_text())["control_points"])
    z, x, y = (np.array(v, float) for v in zip(*pts))
    return lambda zq: (np.interp(zq, z, x), np.interp(zq, z, y))


def load_paths(auto_dir: Path) -> dict[str, dict]:
    info = {}
    for stats in sorted(auto_dir.glob("auto_z*.stats.json")):
        doc = json.loads(stats.read_text())
        for p in doc["paths"]:
            info[p["name"]] = {"length": p["length_px"], "stops": tuple(sorted(p["stop_reasons"]))}
        pcl = json.loads(stats.with_name(stats.name.replace(".stats", "")).read_text())
        for col in pcl["collections"].values():
            pts = list(col["points"].values())
            info[col["name"]]["xyz"] = np.array(pts[len(pts) // 2]["p"])
    return info


def table(title: str, groups: dict[str, list[dict]]) -> None:
    print(f"\n{title}")
    print(f"  {'group':<28}{'paths':>8}{'points ok':>11}{'paths all ok':>14}")
    for key, rows in groups.items():
        pts = sum(r["total_points"] for r in rows)
        ok = sum(r["satisfied_points"] for r in rows)
        full = sum(r["satisfied_points"] == r["total_points"] for r in rows)
        print(f"  {key:<28}{len(rows):>8}{ok / max(pts, 1):>11.3f}{full / max(len(rows), 1):>14.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--satisfied", required=True, type=Path)
    ap.add_argument("--auto-dir", required=True, type=Path)
    ap.add_argument("--umbilicus", required=True, type=Path)
    args = ap.parse_args()
    entries = [e for e in json.loads(args.satisfied.read_text())["pcls"]
               if Path(str(e.get("source_file"))).name == "drawn_control_points.json"]
    info = load_paths(args.auto_dir)
    centre = umbilicus_xy(args.umbilicus)

    by_len, by_radius, by_stop = defaultdict(list), defaultdict(list), defaultdict(list)
    length_bins = [(40, 60), (60, 100), (100, 200), (200, 400), (400, 1e9)]
    radius_bins = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 1e9)]
    for e in entries:
        p = info[e["name"]]
        lo, hi = next(b for b in length_bins if b[0] <= p["length"] < b[1])
        by_len[f"length {lo:.0f}-{hi:.0f} px" if hi < 1e9 else f"length >= {lo:.0f} px"].append(e)
        cx, cy = centre(p["xyz"][2])
        r = float(np.hypot(p["xyz"][0] - cx, p["xyz"][1] - cy))
        lo, hi = next(b for b in radius_bins if b[0] <= r < b[1])
        by_radius[f"radius {lo:.0f}-{hi:.0f} px" if hi < 1e9 else f"radius >= {lo:.0f} px"].append(e)
        by_stop[" + ".join(p["stops"])].append(e)
    table("all automatic paths", {"all": entries})
    table("by path length", dict(sorted(by_len.items(), key=lambda kv: float(kv[0].split()[-2].split("-")[0]))))
    table("by distance from the umbilicus", dict(sorted(
        by_radius.items(), key=lambda kv: float(kv[0].split()[-2].split("-")[0]))))
    common = dict(sorted(by_stop.items(), key=lambda kv: -len(kv[1]))[:8])
    table("by stop reasons (8 most common pairs)", common)


if __name__ == "__main__":
    main()
