#!/usr/bin/env python3
"""Cast adjacency rays between the traced paths of every slice (windcon.relwind.find_rays).

Reads the raw band-centre paths of trace_window.py (auto_z<Z>.npz) and the CT slice they were
traced in; writes adj_z<Z>.npz per slice: for every ray that reached another traced path, the
inner and outer path (indices into that slice's paths), the vertex of each, and the verdict code
(windcon.relwind.VERDICTS). Slices with an output are skipped.

    uv run python experiments/spiral_fit/find_adjacency.py --raw-dir auto_paths_z5 --out adjacency \
        --umbilicus data/PHercParis4/umbilicus.json --bbox 800 7660 1260 7800 --workers 16
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import ndimage

from windcon import evaluate
from windcon.cli import PARIS4_L2
from windcon.relwind import VERDICTS, find_rays
from windcon.tracer import TraceParams
from windcon.zarr_io import ZarrArray

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_surfaces import umbilicus_xy  # noqa: E402

Z_HALF = 2  # as trace_window.py


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def slice_rays(npz: str, url: str, bbox: tuple[int, int, int, int], umbilicus: str, out: str) -> dict:
    t0 = time.time()
    with np.load(npz) as d:
        xy, offsets, z = d["xy"].astype(np.float64), d["offsets"], int(d["z"])
    paths = [xy[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    x0, x1, y0, y1 = bbox
    tp = TraceParams()
    image = evaluate.read_crop(ZarrArray(url), z, x0, x1, y0, y1, Z_HALF)
    smooth = ndimage.gaussian_filter(image, tp.sigma_image)
    center = np.array(umbilicus_xy(Path(umbilicus))(float(z)), dtype=np.float64).ravel()
    rays = find_rays(smooth, np.array([x0, y0], dtype=np.float64), paths, center, tp)
    code = np.array([VERDICTS.index(v) for v in rays.verdict], dtype=np.int8)
    tmp = Path(out) / f"adj_z{z}.npz.part"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, z=np.int64(z), inner=rays.inner, outer=rays.outer,
                            inner_vertex=rays.inner_vertex, outer_vertex=rays.outer_vertex, verdict=code)
    tmp.rename(Path(out) / f"adj_z{z}.npz")
    return {"z": z, "paths": len(paths), "rays": len(code),
            **{v: int((code == i).sum()) for i, v in enumerate(VERDICTS)},
            "pairs": len(rays.pairs(min_votes=3)), "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--umbilicus", required=True, type=Path)
    ap.add_argument("--bbox", type=int, nargs=4, required=True, metavar=("X0", "X1", "Y0", "Y1"))
    ap.add_argument("--volume", default=PARIS4_L2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--z", type=int, nargs="*", help="only these slices")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    todo = [f for f in sorted(args.raw_dir.glob("auto_z*.npz"))
            if not (args.out / f"adj_{f.stem.removeprefix('auto_')}.npz").exists()
            and (not args.z or int(f.stem.removeprefix("auto_z")) in args.z)]
    log(f"{len(todo)} slices with {args.workers} workers")
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(slice_rays, str(f), args.volume, tuple(args.bbox), str(args.umbilicus), str(args.out))
                for f in todo]
        for fut in as_completed(futs):
            log(str(fut.result()))
    log("done")


if __name__ == "__main__":
    sys.exit(main())
