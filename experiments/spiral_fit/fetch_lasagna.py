#!/usr/bin/env python3
"""Mirror the z-rows of a scroll's lasagna stores that a fit window needs.

Only Paris 4 has a spiral-input dataset with packed lasagna sidecars. For the other scrolls the
stores are OME-Zarr folders in the open-data bucket, e.g.
    PHerc0343/representations/predictions/lasagna/20250521140437-lasagna-20260419180421/
with PHerc0343_nx.ome.zarr, _ny and _grad_mag inside. This script copies the chunk rows of one
pyramid group that intersect the window into <out>/, in the stores' own layout, after which
villa packs them:
    python pack_resident_pools.py <out> --what normals,grad_mag --normal-group <group>

    fetch_lasagna.py --lasagna <path in the bucket> --name PHerc0343 --group 2 \
        --z-begin 9500 --z-end 11000 --out <dataset>/lasagna_inputs

Stdlib only. Re-running resumes: a finished z-row is recorded and skipped.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

# CloudFront front of s3://vesuvius-challenge-open-data (see windcon.zarr_io).
MIRROR = "https://data.aws.ash2txt.org/samples"
CHANNELS = ("nx", "ny", "grad_mag")


def http(url: str, retries: int = 6) -> bytes | None:
    """GET a URL; None when the object does not exist (an empty chunk is not stored)."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            if err.code in (403, 404):
                return None
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(min(30, 2**attempt))
    raise AssertionError("unreachable")


def fetch_chunk(url: str, dest: Path) -> int:
    body = http(url)
    if body is None:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(body)
    tmp.rename(dest)
    return len(body)


def mirror_store(url: str, out: Path, group: str, z_begin: int, z_end: int, margin: int, pool) -> None:
    """Copy the metadata of one store and the chunk rows of `group` that cover [z_begin - margin, z_end + margin)."""
    out.mkdir(parents=True, exist_ok=True)
    for name in (".zattrs", ".zgroup"):
        body = http(f"{url}/{name}")
        if body is None:
            raise FileNotFoundError(f"{url}/{name}")
        (out / name).write_bytes(body)
    scales = {
        level["path"]: level["coordinateTransformations"][0]["scale"]
        for level in json.loads((out / ".zattrs").read_text())["multiscales"][0]["datasets"]
    }
    if group not in scales:
        raise SystemExit(f"{url} has groups {sorted(scales)}, not {group}")
    (out / group).mkdir(exist_ok=True)
    meta_body = http(f"{url}/{group}/.zarray")
    (out / group / ".zarray").write_bytes(meta_body)
    meta = json.loads(meta_body)
    shape, chunks = meta["shape"], meta["chunks"]
    separator = meta.get("dimension_separator", ".")
    scale_z = float(scales[group][0])
    first = max(0, int((z_begin - margin) / scale_z) // chunks[0])
    last = min(shape[0] - 1, int((z_end + margin - 1) / scale_z)) // chunks[0]
    done_path = out / group / "rows_done.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    grid = list(product(range(-(-shape[1] // chunks[1])), range(-(-shape[2] // chunks[2]))))
    for row in range(first, last + 1):
        if row in done:
            continue
        started = time.time()
        keys = [separator.join(map(str, (row, y, x))) for y, x in grid]
        sizes = list(pool.map(lambda key: fetch_chunk(f"{url}/{group}/{key}", out / group / key), keys))
        done.add(row)
        done_path.write_text(json.dumps(sorted(done)))
        print(
            f"{time.strftime('%H:%M:%S')} {out.name}/{group} row {row} ({first}-{last}): "
            f"{sum(1 for s in sizes if s):,} of {len(grid):,} chunks, {sum(sizes) / 2**20:.0f} MiB, "
            f"{time.time() - started:.0f} s",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lasagna", required=True, help="the lasagna folder's path in the open-data bucket")
    ap.add_argument("--name", required=True, help="prefix of the stores: <name>_nx.ome.zarr")
    ap.add_argument("--group", required=True, help="pyramid group to copy (normal_zarr_group in spiral-scroll.json)")
    ap.add_argument("--z-begin", type=int, required=True, help="fit window, in voxels of the fitted volume")
    ap.add_argument("--z-end", type=int, required=True)
    ap.add_argument("--margin", type=int, default=160, help="extra voxels at both ends (the flow field's z margin)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    with ThreadPoolExecutor(args.workers) as pool:
        for channel in CHANNELS:
            store = f"{args.name}_{channel}.ome.zarr"
            mirror_store(f"{MIRROR}/{args.lasagna.strip('/')}/{store}", args.out / store, args.group,
                         args.z_begin, args.z_end, args.margin, pool)
    print(time.strftime("%H:%M:%S"), "done", flush=True)


if __name__ == "__main__":
    main()
