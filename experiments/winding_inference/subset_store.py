#!/usr/bin/env python3
"""Write a store with part of the rays of another one.

    subset_store.py <store> <out> --centre-only   # the centre ray of each slab, which is all villa's exporter keeps
    subset_store.py <store> <out> --no-centre     # the other rays of each slab
    subset_store.py <store> <out> --every 2       # every 2nd slab
    subset_store.py <store> <out> --z-range 9900 10900 --count 12000 --seed 0
        # 12,000 rays drawn at random from those fit_spiral.py would use in that z-window

The source is a store from build_store.py or from villa's export_spiral_supervision.py (the
published Paris 4 store). Needs numpy only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from build_store import load_store, save_store

PER_CROSSING = ("crossing_t", "crossing_level")


def reaches(rays: dict[str, np.ndarray], z_begin: float, z_end: float) -> np.ndarray:
    """Rays whose first-to-last crossing spans into [z_begin, z_end): fit_spiral.py samples only these."""
    last = np.cumsum(rays["crossing_counts"]) - 1
    first = last - rays["crossing_counts"] + 1
    has = rays["crossing_counts"] > 0
    z = rays["ray_origin_zyx"][:, 0], rays["ray_step_zyx"][:, 0]
    first_z = z[0] + rays["crossing_t"][np.where(has, first, 0)] * z[1]
    last_z = z[0] + rays["crossing_t"][np.where(has, last, 0)] * z[1]
    return has & (np.maximum(first_z, last_z) >= z_begin) & (np.minimum(first_z, last_z) < z_end)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("store", type=Path)
    ap.add_argument("out", type=Path)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--centre-only", action="store_true")
    group.add_argument("--no-centre", action="store_true")
    ap.add_argument("--every", type=int, default=1, help="keep the rays of every Nth slab")
    ap.add_argument("--z-range", type=float, nargs=2, metavar=("BEGIN", "END"), help="keep rays this window uses")
    ap.add_argument("--count", type=int, help="keep this many of the remaining rays, drawn at random")
    ap.add_argument("--seed", type=int, default=0, help="seed of the random draw")
    args = ap.parse_args()
    rays, manifest = load_store(args.store)
    slab, column_a, column_b = rays["ray_source"].T
    keep = slab % args.every == 0
    centre = manifest.get("columns_per_axis", 1) // 2
    is_centre = (column_a == centre) & (column_b == centre)
    if args.centre_only:
        keep &= is_centre
    if args.no_centre:
        keep &= ~is_centre
    if args.z_range:
        keep &= reaches(rays, *args.z_range)
    if args.count is not None:
        candidates = np.flatnonzero(keep)
        if args.count > len(candidates):
            raise SystemExit(f"only {len(candidates):,} rays are left to draw {args.count:,} from")
        keep = np.zeros_like(keep)
        keep[np.random.default_rng(args.seed).choice(candidates, args.count, replace=False)] = True
    crossing_keep = np.repeat(keep, rays["crossing_counts"])
    subset = {name: rays[name][crossing_keep if name in PER_CROSSING else keep] for name in rays}
    record = {
        key: value
        for key, value in manifest.items()
        if key not in ("artifact_type", "format_version", "coordinate_order", "coordinate_space", "num_rays",
                       "num_crossings", "shards", "fingerprint")
    }  # fmt: skip
    record |= {
        "subset_of": str(args.store),
        "subset_centre_only": args.centre_only,
        "subset_no_centre": args.no_centre,
        "subset_every": args.every,
        "subset_z_range": args.z_range,
        "subset_count": args.count,
        "subset_seed": args.seed if args.count is not None else None,
    }
    args.out.mkdir(parents=True)
    result = save_store(args.out, subset, record)
    print(f"{args.out}: {result['num_rays']:,} of {manifest['num_rays']:,} rays, {result['num_crossings']:,} crossings")


if __name__ == "__main__":
    main()
