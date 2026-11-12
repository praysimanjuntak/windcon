#!/usr/bin/env python3
"""Write a store with part of the rays of one that build_store.py made.

    subset_store.py <store> <out> --centre-only   # the centre ray of each slab, which is all villa's exporter keeps
    subset_store.py <store> <out> --every 2       # every 2nd slab

Needs numpy only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from build_store import load_store, save_store


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("store", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--centre-only", action="store_true")
    ap.add_argument("--every", type=int, default=1, help="keep the rays of every Nth slab")
    args = ap.parse_args()
    rays, manifest = load_store(args.store)
    slab, column_a, column_b = rays["ray_source"].T
    keep = slab % args.every == 0
    if args.centre_only:
        centre = manifest["columns_per_axis"] // 2
        keep &= (column_a == centre) & (column_b == centre)
    crossing_keep = np.repeat(keep, rays["crossing_counts"])
    subset = {
        name: rays[name][crossing_keep if name in ("crossing_t", "crossing_level") else keep] for name in rays
    }
    record = {
        key: value
        for key, value in manifest.items()
        if key not in ("artifact_type", "format_version", "coordinate_order", "coordinate_space", "num_rays",
                       "num_crossings", "shards", "fingerprint")
    }  # fmt: skip
    record |= {"subset_of": str(args.store), "subset_centre_only": args.centre_only, "subset_every": args.every}
    args.out.mkdir(parents=True)
    result = save_store(args.out, subset, record)
    print(f"{args.out}: {result['num_rays']:,} of {manifest['num_rays']:,} rays, {result['num_crossings']:,} crossings")


if __name__ == "__main__":
    main()
