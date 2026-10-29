#!/usr/bin/env python3
"""Trace automatic same-winding paths in whole XY slices of a z-window (PHerc. Paris 4, level 2).

1. prefetch: download every level-2 chunk the slices touch (one process, many threads), so the
   tracing workers below only read the local chunk cache;
2. trace: one worker per slice traces every clean band (windcon.tracer.trace_region) and writes
   ``auto_z<Z>.json`` (VC3D same-winding point collections), ``auto_z<Z>.stats.json`` and
   ``auto_z<Z>.npz`` (the raw 2 px path vertices, for build_chains.py).

Example:
    uv run python experiments/spiral_fit/trace_window.py --out out/auto_paths \
        --bbox 1000 7800 600 7100 --z 9925 9975 10025 --workers 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np

from windcon import evaluate
from windcon.annotations import write_same_winding
from windcon.cli import PARIS4_L2
from windcon.render import region_png
from windcon.tracer import SliceField, TraceParams, trace_region
from windcon.zarr_io import ZarrArray

Z_HALF = 2
SPACING = 20.0
MIN_LENGTH = 40.0


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def prefetch(vol: ZarrArray, zs: list[int], bbox: tuple[int, int, int, int], threads: int) -> None:
    x0, x1, y0, y1 = bbox
    cz, cy, cx = vol.chunks
    z_rows = sorted({z // cz for z0 in zs for z in range(z0 - Z_HALF, z0 + Z_HALF + 1)})
    idx = list(product(z_rows, range(y0 // cy, (y1 - 1) // cy + 1), range(x0 // cx, (x1 - 1) // cx + 1)))
    log(f"prefetch: {len(idx)} chunks in z-rows {z_rows}")
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(threads) as pool:
        for _ in as_completed([pool.submit(vol._fetch, i) for i in idx]):
            done += 1
            if done % 2000 == 0:
                log(f"prefetch: {done}/{len(idx)} ({time.time() - t0:.0f}s)")
    present = sum(1 for i in idx if vol._chunk_path(i).exists())
    log(f"prefetch: done, {present} chunks present, {len(idx) - present} empty ({time.time() - t0:.0f}s)")


def trace_slice(url: str, z: int, bbox: tuple[int, int, int, int], out_dir: str, png: bool) -> dict:
    t0 = time.time()
    vol = ZarrArray(url)
    x0, x1, y0, y1 = bbox
    params = TraceParams()
    image = evaluate.read_crop(vol, z, x0, x1, y0, y1, Z_HALF)
    paths = trace_region(SliceField(image, params), min_length=MIN_LENGTH)
    origin = np.array([x0, y0], dtype=np.float64)
    strips, stats = [], []
    for i, path in enumerate(paths):
        xy = path.resampled(SPACING) + origin
        name = f"auto_z{z}_{i:05d}"
        strips.append((name, np.column_stack([xy, np.full(len(xy), float(z))])))
        stats.append({"name": name, "length_px": round(path.length, 1),
                      "stop_reasons": list(path.stop_reasons)})
    out = Path(out_dir) / f"auto_z{z}.json"
    write_same_winding(out, strips)
    out.with_suffix(".stats.json").write_text(json.dumps(
        {"volume": url, "z": z, "box_x0_x1_y0_y1": list(bbox), "z_half": Z_HALF,
         "spacing_px": SPACING, "min_length_px": MIN_LENGTH, "paths": stats}))
    if png:
        step = 4  # preview at 1/4 resolution
        small = image[::step, ::step]
        region_png(small, (0, 0), [p.xy / step for p in paths], Path(out_dir) / f"auto_z{z}.png",
                   title=f"z={z}: {len(paths)} paths (1/{step} scale)")
    # Raw 2 px vertices (volume x, y) for build_chains.py. Written last and atomically: it marks the
    # slice as complete, so an interrupted run redoes exactly the unfinished slices.
    tmp = out.with_suffix(".npz.part")
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            xy=np.concatenate([p.xy + origin for p in paths]).astype(np.float32) if paths else np.zeros((0, 2), np.float32),
            offsets=np.cumsum([0] + [len(p.xy) for p in paths]).astype(np.int64),
            z=np.int64(z),
        )
    tmp.rename(out.with_suffix(".npz"))
    return {"z": z, "paths": len(paths), "length_px": round(sum(p.length for p in paths)),
            "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--z", type=int, nargs="+", required=True)
    ap.add_argument("--bbox", type=int, nargs=4, required=True, metavar=("X0", "X1", "Y0", "Y1"))
    ap.add_argument("--volume", default=PARIS4_L2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=96, help="download threads for the prefetch")
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    bbox = tuple(args.bbox)
    prefetch(ZarrArray(args.volume), args.z, bbox, args.threads)
    todo = [z for z in args.z if not (args.out / f"auto_z{z}.npz").exists()]
    log(f"trace: {len(todo)} slices with {args.workers} workers")
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(trace_slice, args.volume, z, bbox, str(args.out), args.png) for z in todo]
        for fut in as_completed(futs):
            log(f"trace: {fut.result()}")
    log("done")


if __name__ == "__main__":
    sys.exit(main())
