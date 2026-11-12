#!/usr/bin/env python3
"""Render the layer stack around a tifxyz mesh from the scan, as one TIFF per layer.

A tifxyz folder holds x.tif, y.tif and z.tif: the position of every node of a grid, in voxels of
the scan's level 0, and -1 where there is no node. meta.json's `scale` is nodes per output pixel
(0.05: one node every 20 pixels). Positions and normals are interpolated linearly between nodes.
Layer k of N samples the scan (trilinear) at k - (N - 1) / 2 voxels along the surface normal, as
VC3D's `vc_render_tifxyz --scale 1 --slice-step 1` does. With --umbilicus the normals are turned
away from the scroll's axis; without it they are the cross product of the grid's column and row
directions.

    uv run python render_mesh.py <mesh.tifxyz> --volume <URL of the scan's level 0> \
        --umbilicus <umbilicus.json> --layers 66 --out <folder>/<name>_layers

The result is the TIFF-folder input of fetch_layers.py:
    uv run python fetch_layers.py <folder>/<name>_layers --name <name> --native-spacing 8.64 ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from windcon.zarr_io import ZarrArray  # noqa: E402


def load_mesh(folder: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Node positions [rows, cols, 3] as (x, y, z), the mask of real nodes, and nodes per pixel."""
    from PIL import Image

    xyz = np.stack([np.asarray(Image.open(folder / f"{axis}.tif"), dtype=np.float64) for axis in "xyz"], axis=-1)
    scale = json.loads((folder / "meta.json").read_text())["scale"]
    if scale[0] != scale[1]:
        raise SystemExit(f"anisotropic grid scale {scale}")
    return xyz, (xyz > 0).all(-1), float(scale[0])


def node_normals(xyz: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Unit normals at the nodes: column direction x row direction, from the nearest valid neighbours."""

    def difference(axis: int) -> np.ndarray:
        ahead = np.roll(xyz, -1, axis=axis)
        behind = np.roll(xyz, 1, axis=axis)
        has_ahead = np.roll(valid, -1, axis=axis)
        has_behind = np.roll(valid, 1, axis=axis)
        edge = [slice(None)] * 2
        edge[axis] = -1
        has_ahead[tuple(edge)] = False
        edge[axis] = 0
        has_behind[tuple(edge)] = False
        upper = np.where(has_ahead[..., None], ahead, xyz)
        lower = np.where(has_behind[..., None], behind, xyz)
        return upper - lower

    normal = np.cross(difference(1), difference(0))
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    return np.where(length > 0, normal / np.maximum(length, 1e-12), 0.0)


def umbilicus_xy(path: Path, z: np.ndarray) -> np.ndarray:
    """Axis position (x, y) at the given z, interpolated between the control points."""
    points = sorted((p["z"], p["x"], p["y"]) for p in json.loads(path.read_text())["control_points"])
    zs, xs, ys = (np.array(column, dtype=np.float64) for column in zip(*points))
    return np.stack([np.interp(z, zs, xs), np.interp(z, zs, ys)], axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mesh", type=Path)
    ap.add_argument("--volume", required=True, help="URL of the scan's level-0 array")
    ap.add_argument("--umbilicus", type=Path, help="turn the normals away from this axis")
    ap.add_argument("--layers", type=int, default=66)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tile", type=int, default=256, help="pixels per side of a sampling tile")
    args = ap.parse_args()
    from PIL import Image

    xyz, valid, scale = load_mesh(args.mesh)
    normals = node_normals(xyz, valid)
    if args.umbilicus:
        outward = xyz[..., :2] - umbilicus_xy(args.umbilicus, xyz[..., 2])
        flip = (normals[..., :2] * outward).sum(-1) < 0
        normals[flip] *= -1
        print(f"normals turned away from the axis at {flip[valid].mean():.1%} of the nodes")
    step = round(1 / scale)
    height, width = xyz.shape[0] * step, xyz.shape[1] * step
    offsets = np.arange(args.layers) - (args.layers - 1) / 2
    volume = ZarrArray(args.volume)
    stack = np.zeros((args.layers, height, width), dtype=np.uint8)
    started = time.time()
    tiles = [(r, c) for r in range(0, height, args.tile) for c in range(0, width, args.tile)]
    for count, (r0, c0) in enumerate(tiles, 1):
        rows = np.arange(r0, min(r0 + args.tile, height)) * scale
        cols = np.arange(c0, min(c0 + args.tile, width)) * scale
        grid = np.meshgrid(rows, cols, indexing="ij")
        inside = ndimage.map_coordinates(valid.astype(np.float64), grid, order=1, mode="nearest") > 0.999
        if not inside.any():
            continue
        # Holes hold -1; a pixel next to one is dropped by `inside`, so its interpolated value is never used.
        position = np.stack([ndimage.map_coordinates(xyz[..., a], grid, order=1, mode="nearest") for a in range(3)], -1)
        normal = np.stack([ndimage.map_coordinates(normals[..., a], grid, order=1, mode="nearest") for a in range(3)], -1)
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
        points = position[None] + offsets[:, None, None, None] * normal[None]  # [layers, h, w, (x, y, z)]
        used = points[:, inside]
        low = np.floor(used.reshape(-1, 3).min(0)).astype(int)
        high = np.floor(used.reshape(-1, 3).max(0)).astype(int) + 2
        block = volume.read(((low[2], high[2]), (low[1], high[1]), (low[0], high[0]))).astype(np.float32)
        values = ndimage.map_coordinates(block, [used[..., 2] - low[2], used[..., 1] - low[1], used[..., 0] - low[0]],
                                         order=1, mode="constant", cval=0.0)
        tile = np.zeros(points.shape[:3], dtype=np.uint8)
        tile[:, inside] = np.clip(values, 0, 255).astype(np.uint8)
        stack[:, r0 : r0 + tile.shape[1], c0 : c0 + tile.shape[2]] = tile
        if count % 20 == 0 or count == len(tiles):
            print(f"{time.strftime('%H:%M:%S')} {count}/{len(tiles)} tiles, {time.time() - started:.0f} s", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for index, layer in enumerate(stack):
        Image.fromarray(layer).save(args.out / f"{index:02d}.tif")
    print(f"{args.out}: {args.layers} layers of {height} x {width}, {(stack[args.layers // 2] > 0).mean():.1%} covered")


if __name__ == "__main__":
    main()
